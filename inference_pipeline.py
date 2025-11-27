"""
License Plate OCR Inference Pipeline

Kế thừa từ base class InferenceModel theo format chuẩn.
"""

import os
import yaml
import numpy as np
import cv2
from typing import Any, Dict, List, Union, Optional

from fast_plate_ocr import LicensePlateRecognizer
from preprocessing import PreprocessingPipeline


class InferenceModel:
    """Base class cho inference pipeline."""

    def __init__(self, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Lỗi: File mô hình không tìm thấy tại đường dẫn: {model_path}")
        
        self.model_path = model_path
        self.device = device
        self.model = None
        print(f"InferenceModel được khởi tạo với đường dẫn mô hình: '{self.model_path}' trên thiết bị: '{self.device}'.")

    def load_model(self) -> Any:
        raise NotImplementedError("Phương thức 'load_model' cần được triển khai.")

    def preprocess(self, data: Any) -> Any:
        raise NotImplementedError("Phương thức 'preprocess' cần được triển khai.")

    def infer(self, preprocessed_data: Any) -> Any:
        raise NotImplementedError("Phương thức 'infer' cần được triển khai.")

    def postprocess(self, model_output: Any) -> Any:
        raise NotImplementedError("Phương thức 'postprocess' cần được triển khai.")

    def run_inference(self, raw_data: Any) -> Any:
        print("Bắt đầu tiền xử lý dữ liệu...")
        preprocessed_data = self.preprocess(raw_data)
        print("Tiền xử lý hoàn tất. Bắt đầu suy luận...")
        model_output = self.infer(preprocessed_data)
        print("Suy luận hoàn tất. Bắt đầu hậu xử lý...")
        final_result = self.postprocess(model_output)
        print("Hậu xử lý hoàn tất.")
        return final_result


class LicensePlateOCRPipeline(InferenceModel):
    """
    Pipeline OCR biển số xe sử dụng fast-plate-ocr.
    
    Example:
        pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")
        result = pipeline.run_inference("plate.jpg")
        results = pipeline.run_inference(["plate1.jpg", "plate2.jpg"])
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cpu",
        preprocessing_config: Optional[Dict[str, Any]] = None,
        postprocessing_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 8,
        providers: Optional[List[str]] = None
    ):
        super().__init__(model_path, device)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config không tìm thấy: {config_path}")
        
        self.config_path = config_path
        self.batch_size = batch_size
        self.providers = providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" 
            else ["CPUExecutionProvider"]
        )
        
        self.preprocessing_config = preprocessing_config or {}
        self.preprocessor = PreprocessingPipeline(self.preprocessing_config)
        
        self.postprocessing_config = postprocessing_config or {
            "normalize": {
                "remove_underscore": True,
                "remove_hyphen": True,
                "remove_dot": True,
                "remove_space": True,
                "to_uppercase": True
            }
        }

    @classmethod
    def from_config(cls, config_path: str) -> "LicensePlateOCRPipeline":
        """Factory method: Tạo pipeline từ file config YAML."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file không tìm thấy: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        model_cfg = config.get("model", {})
        
        return cls(
            model_path=model_cfg.get("onnx_model_path"),
            config_path=model_cfg.get("plate_config_path"),
            device=model_cfg.get("device", "cpu"),
            preprocessing_config=config.get("preprocessing", {}),
            postprocessing_config=config.get("postprocessing", {}),
            batch_size=config.get("inference", {}).get("batch_size", 8),
            providers=model_cfg.get("providers")
        )

    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    
    def load_model(self) -> LicensePlateRecognizer:
        print(f"Đang tải mô hình từ {self.model_path}...")
        self.model = LicensePlateRecognizer(
            onnx_model_path=self.model_path,
            plate_config_path=self.config_path,
            providers=self.providers
        )
        print("Mô hình đã tải thành công.")
        return self.model

    # =========================================================================
    # PREPROCESS
    # =========================================================================
    
    def preprocess(self, data: Union[str, np.ndarray, List]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(data, list):
            return [self.preprocess(item) for item in data]

        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Image không tìm thấy: {data}")
            image = cv2.imread(data)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {data}")
        elif isinstance(data, np.ndarray):
            image = data
        else:
            raise TypeError(f"Input phải là str hoặc np.ndarray, nhận được: {type(data)}")

        return self.preprocessor.process(image)

    # =========================================================================
    # INFER
    # =========================================================================
    
    def infer(self, preprocessed_data: Union[np.ndarray, List[np.ndarray]]) -> Union[str, List[str]]:
        """Thực hiện inference - tự động detect single hay batch."""
        if self.model is None:
            raise ValueError("Lỗi: Mô hình chưa được tải. Vui lòng gọi 'load_model()' trước.")
        
        # Single inference
        if isinstance(preprocessed_data, np.ndarray):
            result = self.model.run(preprocessed_data)
            return result[0] if isinstance(result, list) and result else ""
        
        # Batch inference
        if not preprocessed_data:
            return []
        
        results = self.model.run(preprocessed_data)
        return results if isinstance(results, list) else [results]

    # =========================================================================
    # POSTPROCESS
    # =========================================================================
    
    def postprocess(self, model_output: Union[str, List[str]]) -> Union[str, List[str]]:
        """Hậu xử lý: chuẩn hóa text."""
        if isinstance(model_output, list):
            return [self._normalize_text(text) for text in model_output]
        return self._normalize_text(model_output)
    
    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa text theo config."""
        if not text:
            return ""
        
        cfg = self.postprocessing_config.get("normalize", {})
        result = text.upper() if cfg.get("to_uppercase", True) else text
        
        for char, key in [("_", "remove_underscore"), ("-", "remove_hyphen"), 
                          (".", "remove_dot"), (" ", "remove_space")]:
            if cfg.get(key, True):
                result = result.replace(char, "")
        return result

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def enable_preprocessing_step(self, step: str, enabled: bool = True):
        """Bật/tắt một bước preprocessing."""
        if step not in self.preprocessing_config:
            self.preprocessing_config[step] = {}
        self.preprocessing_config[step]["enabled"] = enabled
        self.preprocessor = PreprocessingPipeline(self.preprocessing_config)
        print(f"Step '{step}' {'enabled' if enabled else 'disabled'}.")

    def get_raw_prediction(self, data: Any) -> Union[str, List[str]]:
        """Lấy prediction thô (không postprocess)."""
        if self.model is None:
            self.load_model()
        return self.infer(self.preprocess(data))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    CONFIG_PATH = "pipeline_config.yaml"
    
    try:
        pipeline = LicensePlateOCRPipeline.from_config(CONFIG_PATH)
        pipeline.load_model()
        
        # Tắt preprocessing
        for step in ["upscale", "correct_skew", "denoise", "enhance_contrast", "sharpen"]:
            pipeline.enable_preprocessing_step(step, False)
        
        # Test inference
        result = pipeline.run_inference(["./data/cam2_20250926_133219_obj03_cls1_lp00_c084.jpg","./data/cam1_20250926_143554_car_0_plate_0_0.jpg", "./data/cam3_20251001_112631_obj04_cls1_lp00_c083.jpg"])
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")