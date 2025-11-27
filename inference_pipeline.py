"""
License Plate OCR Inference Pipeline

Pipeline inference cho OCR biển số xe, kế thừa từ base class InferenceModel.
Hỗ trợ:
- Single và Batch inference
- Cấu hình preprocessing on/off từng bước qua file YAML
- Nhiều loại input: path, numpy array, list of paths, list of arrays
"""

import os
import gc
import yaml
import time
import json
import csv
import numpy as np
import cv2
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, field, asdict

from fast_plate_ocr import LicensePlateRecognizer
from preprocessing import PreprocessingPipeline


# =============================================================================
# BENCHMARK DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Kết quả benchmark cho một cấu hình - chỉ các metrics quan trọng"""
    batch_size: int
    n_runs: int
    n_images: int
    
    # Latency (ms) - thời gian xử lý 1 batch
    latency_avg: float = 0.0
    latency_std: float = 0.0
    
    # Throughput
    fps: float = 0.0  # images per second
    
    # Memory (MB)
    ram_peak: float = 0.0  # RAM peak trong quá trình inference


@dataclass 
class BenchmarkReport:
    """Báo cáo benchmark tổng hợp"""
    model_path: str
    device: str
    preprocessing_steps: List[str]
    timestamp: str = ""
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "preprocessing_steps": self.preprocessing_steps,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results]
        }


# =============================================================================
# BASE CLASS 
# =============================================================================

class InferenceModel:
    """
    Base class cho inference pipeline.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Khởi tạo InferenceModel.

        Args:
            model_path: Đường dẫn đến file mô hình
            device: Thiết bị chạy model ('cpu' hoặc 'cuda')
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model không tìm thấy: {model_path}")

        self.model_path = model_path
        self.device = device
        self.model = None

    def load_model(self) -> Any:
        """Tải model - cần implement ở class con"""
        raise NotImplementedError("Cần implement load_model()")

    def preprocess(self, data: Any) -> Any:
        """Tiền xử lý dữ liệu - cần implement ở class con"""
        raise NotImplementedError("Cần implement preprocess()")

    def infer(self, preprocessed_data: Any) -> Any:
        """Thực hiện inference - cần implement ở class con"""
        raise NotImplementedError("Cần implement infer()")

    def postprocess(self, model_output: Any) -> Any:
        """Hậu xử lý kết quả - cần implement ở class con"""
        raise NotImplementedError("Cần implement postprocess()")

    def run_inference(self, raw_data: Any) -> Any:
        """
        Chạy toàn bộ pipeline: preprocess -> infer -> postprocess
        """
        preprocessed_data = self.preprocess(raw_data)
        model_output = self.infer(preprocessed_data)
        final_result = self.postprocess(model_output)
        return final_result


# =============================================================================
# LICENSE PLATE OCR PIPELINE
# =============================================================================

class LicensePlateOCRPipeline(InferenceModel):
    """
    Pipeline OCR biển số xe sử dụng fast-plate-ocr.
    
    Hỗ trợ:
    - Single inference: infer_single()
    - Batch inference: infer_batch()
    - Cấu hình preprocessing qua YAML config
    - Nhiều loại input: str path, numpy array, list
    - Benchmark với metrics chính xác
    
    Example:
        ```python
        # Khởi tạo từ config file
        pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")
        
        # Single inference
        result = pipeline.run_inference("path/to/plate.jpg")
        
        # Batch inference
        results = pipeline.run_inference(["path1.jpg", "path2.jpg", "path3.jpg"])
        
        # Với numpy array
        image = cv2.imread("plate.jpg")
        result = pipeline.run_inference(image)
        ```
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda",
        preprocessing_config: Optional[Dict[str, Any]] = None,
        postprocessing_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 8,
        providers: Optional[List[str]] = None
    ):
        """
        Khởi tạo LicensePlateOCRPipeline.

        Args:
            model_path: Đường dẫn đến ONNX model
            config_path: Đường dẫn đến plate config YAML (của fast-plate-ocr)
            device: 'cuda' hoặc 'cpu'
            preprocessing_config: Config cho preprocessing (từ YAML)
            postprocessing_config: Config cho postprocessing (từ YAML)
            batch_size: Batch size cho batch inference
            providers: ONNX Runtime providers (auto nếu None)
        """
        super().__init__(model_path, device)
        
        self.config_path = config_path
        self.batch_size = batch_size
        
        # Validate config path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config không tìm thấy: {config_path}")
        
        # Setup providers
        if providers is None:
            if device == "cuda":
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = providers
        
        # Setup preprocessing pipeline
        self.preprocessing_config = preprocessing_config or {}
        self.preprocessor = PreprocessingPipeline(self.preprocessing_config)
        
        # Setup postprocessing config
        self.postprocessing_config = postprocessing_config or {
            "normalize": {
                "remove_underscore": True,
                "remove_hyphen": True,
                "remove_dot": True,
                "remove_space": True,
                "to_uppercase": True
            }
        }
        
        print(f"LicensePlateOCRPipeline initialized:")
        print(f"  - Model: {self.model_path}")
        print(f"  - Config: {self.config_path}")
        print(f"  - Device: {self.device}")
        print(f"  - Providers: {self.providers}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Preprocessing steps: {self.preprocessor.get_enabled_steps()}")

    @classmethod
    def from_config(cls, config_path: str) -> "LicensePlateOCRPipeline":
        """
        Factory method: Tạo pipeline từ file config YAML.

        Args:
            config_path: Đường dẫn đến pipeline_config.yaml

        Returns:
            LicensePlateOCRPipeline instance
            
        Example:
            ```python
            pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")
            ```
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file không tìm thấy: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        model_cfg = config.get("model", {})
        preprocessing_cfg = config.get("preprocessing", {})
        postprocessing_cfg = config.get("postprocessing", {})
        inference_cfg = config.get("inference", {})
        
        return cls(
            model_path=model_cfg.get("onnx_model_path"),
            config_path=model_cfg.get("plate_config_path"),
            device=model_cfg.get("device", "cuda"),
            preprocessing_config=preprocessing_cfg,
            postprocessing_config=postprocessing_cfg,
            batch_size=inference_cfg.get("batch_size", 8),
            providers=model_cfg.get("providers")
        )

    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    
    def load_model(self) -> LicensePlateRecognizer:
        """
        Tải model OCR sử dụng fast-plate-ocr LicensePlateRecognizer.

        Returns:
            LicensePlateRecognizer instance
        """
        print(f"Loading model from {self.model_path}...")
        
        try:
            self.model = LicensePlateRecognizer(
                onnx_model_path=self.model_path,
                plate_config_path=self.config_path,
                providers=self.providers
            )
            print(f"✅ Model loaded successfully with providers: {self.providers}")
            return self.model
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # =========================================================================
    # PREPROCESS
    # =========================================================================
    
    def _load_image(self, data: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load ảnh từ path hoặc trả về numpy array.

        Args:
            data: Đường dẫn ảnh (str) hoặc numpy array

        Returns:
            numpy array (BGR)
        """
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Image không tìm thấy: {data}")
            image = cv2.imread(data)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {data}")
            return image
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Input phải là str hoặc np.ndarray, nhận được: {type(data)}")

    def preprocess(
        self,
        data: Union[str, np.ndarray, List[str], List[np.ndarray]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Tiền xử lý dữ liệu đầu vào.
        
        Hỗ trợ:
        - Single path (str)
        - Single numpy array
        - List of paths
        - List of numpy arrays

        Args:
            data: Dữ liệu đầu vào (single hoặc batch)

        Returns:
            Dữ liệu đã tiền xử lý (single np.ndarray hoặc list of np.ndarray)
        """
        # Check if batch input
        is_batch = isinstance(data, list)
        
        if is_batch:
            # Batch preprocessing
            results = []
            for item in data:
                image = self._load_image(item)
                processed = self.preprocessor.process(image)
                results.append(processed)
            return results
        else:
            # Single preprocessing
            image = self._load_image(data)
            return self.preprocessor.process(image)

    # =========================================================================
    # INFERENCE
    # =========================================================================
    
    def infer(
        self,
        preprocessed_data: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[str, List[str]]:
        """
        Thực hiện inference với model.
        Tự động chọn single hoặc batch dựa trên input.

        Args:
            preprocessed_data: Dữ liệu đã tiền xử lý

        Returns:
            Raw predictions (string hoặc list of strings)
        """
        if self.model is None:
            raise ValueError("Model chưa được load. Gọi load_model() trước.")
        
        is_batch = isinstance(preprocessed_data, list)
        
        if is_batch:
            return self.infer_batch(preprocessed_data)
        else:
            return self.infer_single(preprocessed_data)

    def infer_single(self, image: np.ndarray) -> str:
        """
        Single image inference.

        Args:
            image: Single preprocessed image (numpy array)

        Returns:
            Raw prediction string
        """
        if self.model is None:
            raise ValueError("Model chưa được load. Gọi load_model() trước.")
        
        result = self.model.run(image)
        
        # fast-plate-ocr trả về list
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return ""

    def infer_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Batch inference.

        Args:
            images: List of preprocessed images

        Returns:
            List of raw prediction strings
        """
        if self.model is None:
            raise ValueError("Model chưa được load. Gọi load_model() trước.")
        
        if not images:
            return []
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Filter None values
            valid_batch = [img for img in batch if img is not None]
            
            if not valid_batch:
                predictions.extend([""] * len(batch))
                continue
            
            # Run batch inference
            results = self.model.run(valid_batch)
            
            if not isinstance(results, list):
                results = [results]
            
            predictions.extend(results)
        
        return predictions

    # =========================================================================
    # POSTPROCESS
    # =========================================================================
    
    def _normalize_text(self, text: str) -> str:
        """
        Chuẩn hóa text theo config.

        Args:
            text: Raw text từ model

        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        result = text
        normalize_cfg = self.postprocessing_config.get("normalize", {})
        
        if normalize_cfg.get("to_uppercase", True):
            result = result.upper()
        
        if normalize_cfg.get("remove_underscore", True):
            result = result.replace("_", "")
        
        if normalize_cfg.get("remove_hyphen", True):
            result = result.replace("-", "")
        
        if normalize_cfg.get("remove_dot", True):
            result = result.replace(".", "")
        
        if normalize_cfg.get("remove_space", True):
            result = result.replace(" ", "")
        
        return result

    def postprocess(
        self,
        model_output: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """
        Hậu xử lý kết quả từ model.

        Args:
            model_output: Raw output từ model (single string hoặc list)

        Returns:
            Normalized output
        """
        if isinstance(model_output, list):
            return [self._normalize_text(text) for text in model_output]
        else:
            return self._normalize_text(model_output)

    # =========================================================================
    # RUN INFERENCE (Override để hỗ trợ cả single và batch)
    # =========================================================================
    
    def run_inference(
        self,
        raw_data: Union[str, np.ndarray, List[str], List[np.ndarray]]
    ) -> Union[str, List[str]]:
        """
        Chạy toàn bộ pipeline: preprocess -> infer -> postprocess.
        
        Tự động detect single hoặc batch dựa trên input type.

        Args:
            raw_data: Input data
                - str: Single image path
                - np.ndarray: Single image array
                - List[str]: Batch of image paths
                - List[np.ndarray]: Batch of image arrays

        Returns:
            - str: Nếu input là single
            - List[str]: Nếu input là batch
            
        Example:
            ```python
            # Single
            result = pipeline.run_inference("plate.jpg")
            print(result)  # "51G12345"
            
            # Batch
            results = pipeline.run_inference(["plate1.jpg", "plate2.jpg"])
            print(results)  # ["51G12345", "30A67890"]
            ```
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Run pipeline
        preprocessed = self.preprocess(raw_data)
        raw_output = self.infer(preprocessed)
        final_output = self.postprocess(raw_output)
        
        return final_output

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def run_single(
        self,
        data: Union[str, np.ndarray]
    ) -> str:
        """
        Convenience method: Chạy single inference.

        Args:
            data: Single image (path hoặc array)

        Returns:
            Normalized prediction string
        """
        if self.model is None:
            self.load_model()
        
        preprocessed = self.preprocess(data)
        raw_output = self.infer_single(preprocessed)
        return self.postprocess(raw_output)

    def run_batch(
        self,
        data: Union[List[str], List[np.ndarray]]
    ) -> List[str]:
        """
        Convenience method: Chạy batch inference.

        Args:
            data: List of images (paths hoặc arrays)

        Returns:
            List of normalized prediction strings
        """
        if self.model is None:
            self.load_model()
        
        preprocessed = self.preprocess(data)
        raw_output = self.infer_batch(preprocessed)
        return self.postprocess(raw_output)

    def get_raw_prediction(
        self,
        data: Union[str, np.ndarray, List[str], List[np.ndarray]]
    ) -> Union[str, List[str]]:
        """
        Lấy prediction thô (không postprocess).

        Args:
            data: Input data

        Returns:
            Raw predictions (chưa normalize)
        """
        if self.model is None:
            self.load_model()
        
        preprocessed = self.preprocess(data)
        return self.infer(preprocessed)

    def update_preprocessing_config(self, config: Dict[str, Any]):
        """
        Cập nhật config preprocessing runtime.

        Args:
            config: New preprocessing config
        """
        self.preprocessing_config.update(config)
        self.preprocessor = PreprocessingPipeline(self.preprocessing_config)
        print(f"Preprocessing config updated. Enabled steps: {self.preprocessor.get_enabled_steps()}")

    def enable_preprocessing_step(self, step: str, enabled: bool = True):
        """
        Bật/tắt một bước preprocessing cụ thể.

        Args:
            step: Tên bước ("upscale", "denoise", "correct_skew", "enhance_contrast", "sharpen")
            enabled: True để bật, False để tắt
        """
        if step not in self.preprocessing_config:
            self.preprocessing_config[step] = {}
        
        self.preprocessing_config[step]["enabled"] = enabled
        self.preprocessor = PreprocessingPipeline(self.preprocessing_config)
        print(f"Step '{step}' {'enabled' if enabled else 'disabled'}. "
              f"Current steps: {self.preprocessor.get_enabled_steps()}")

    def __repr__(self) -> str:
        return (
            f"LicensePlateOCRPipeline(\n"
            f"  model_path='{self.model_path}',\n"
            f"  config_path='{self.config_path}',\n"
            f"  device='{self.device}',\n"
            f"  batch_size={self.batch_size},\n"
            f"  preprocessing_steps={self.preprocessor.get_enabled_steps()}\n"
            f")"
        )

    # =========================================================================
    # BENCHMARK METHODS - CHỈ GIỮ CÁC METRICS QUAN TRỌNG
    # =========================================================================
    
    def get_process_memory_mb(self) -> float:
        """
        Lấy RSS (Resident Set Size) memory của process hiện tại.
        
        Returns:
            float: Memory usage in MB
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def benchmark_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int,
        n_runs: int = 50,
        warmup_runs: int = 5
    ) -> BenchmarkResult:
        """
        Benchmark batch inference với batch size cụ thể.
        
        Args:
            images: List ảnh test
            batch_size: Batch size để test
            n_runs: Số lần chạy
            warmup_runs: Số lần warmup
            
        Returns:
            BenchmarkResult với metrics: batch_size, latency, fps, ram_peak
        """
        if self.model is None:
            self.load_model()
        
        # Ensure we have enough images
        if len(images) < batch_size:
            images = (images * (batch_size // len(images) + 1))[:batch_size]
        else:
            images = images[:batch_size]
        
        # Force garbage collection
        gc.collect()
        
        # Preprocess all images once
        preprocessed = self.preprocess(images)
        
        # Save and set batch size
        old_batch_size = self.batch_size
        self.batch_size = batch_size
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.infer_batch(preprocessed)
        
        # Force GC sau warmup
        gc.collect()
        
        # Benchmark runs
        latencies = []
        ram_samples = []
        
        for i in range(n_runs):
            # Sample RAM trước mỗi run
            ram_samples.append(self.get_process_memory_mb())
            
            start = time.perf_counter()
            _ = self.infer_batch(preprocessed)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        # Sample RAM cuối
        ram_samples.append(self.get_process_memory_mb())
        
        # Restore batch size
        self.batch_size = old_batch_size
        
        # RAM peak
        ram_peak = max(ram_samples)
        
        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = float(np.mean(latencies))
        
        result = BenchmarkResult(
            batch_size=batch_size,
            n_runs=n_runs,
            n_images=batch_size,
            latency_avg=avg_latency,
            latency_std=float(np.std(latencies)),
            fps=batch_size * 1000.0 / avg_latency,
            ram_peak=ram_peak
        )
        
        return result

    def benchmark_batch_sizes(
        self,
        images: List[Union[str, np.ndarray]],
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        n_runs: int = 50,
        warmup_runs: int = 5,
        verbose: bool = True
    ) -> BenchmarkReport:
        """
        Benchmark nhiều batch sizes để so sánh.
        
        Args:
            images: List ảnh test (nên có >= max(batch_sizes) ảnh)
            batch_sizes: Danh sách batch sizes cần test
            n_runs: Số lần chạy mỗi batch size
            warmup_runs: Số lần warmup
            verbose: In kết quả ra console
            
        Returns:
            BenchmarkReport với tất cả kết quả
        """
        from datetime import datetime
        
        if self.model is None:
            self.load_model()
        
        report = BenchmarkReport(
            model_path=self.model_path,
            device=self.device,
            preprocessing_steps=self.preprocessor.get_enabled_steps(),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        if verbose:
            print("=" * 70)
            print("BENCHMARK: Batch Size Comparison")
            print("=" * 70)
            print(f"Model: {self.model_path}")
            print(f"Device: {self.device}")
            print(f"Preprocessing: {self.preprocessor.get_enabled_steps()}")
            print(f"N runs per batch: {n_runs}")
            print(f"Warmup runs: {warmup_runs}")
            print("=" * 70)
            print(f"{'Batch':<10} {'Latency (ms)':<20} {'FPS':<15} {'RAM Peak':<15}")
            print(f"{'Size':<10} {'avg ± std':<20} {'img/s':<15} {'(MB)':<15}")
            print("-" * 70)
        
        for batch_size in batch_sizes:
            # Force GC trước mỗi benchmark
            gc.collect()
            
            result = self.benchmark_batch(
                images=images,
                batch_size=batch_size,
                n_runs=n_runs,
                warmup_runs=warmup_runs
            )
            report.results.append(result)
            
            if verbose:
                latency_str = f"{result.latency_avg:.2f} ± {result.latency_std:.2f}"
                print(
                    f"{batch_size:<10} "
                    f"{latency_str:<20} "
                    f"{result.fps:<15.2f} "
                    f"{result.ram_peak:<15.2f}"
                )
        
        if verbose:
            print("=" * 70)
            
            # Find optimal batch size
            best_fps = max(report.results, key=lambda x: x.fps)
            best_latency = min(report.results, key=lambda x: x.latency_avg)
            best_ram = min(report.results, key=lambda x: x.ram_peak)
            
            print(f"📊 Best FPS: batch_size={best_fps.batch_size} ({best_fps.fps:.2f} img/s)")
            print(f"📊 Best Latency: batch_size={best_latency.batch_size} ({best_latency.latency_avg:.2f} ms)")
            print(f"📊 Lowest RAM: batch_size={best_ram.batch_size} ({best_ram.ram_peak:.2f} MB)")
            print("=" * 70)
        
        return report

    def export_benchmark_report(
        self,
        report: BenchmarkReport,
        output_path: str,
        format: str = "csv"
    ) -> str:
        """
        Xuất báo cáo benchmark ra file.
        
        Args:
            report: BenchmarkReport từ benchmark_batch_sizes()
            output_path: Đường dẫn file output
            format: "csv" hoặc "json"
            
        Returns:
            Đường dẫn file đã lưu
        """
        format = format.lower()
        
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Header - chỉ các metrics quan trọng
                writer.writerow([
                    "batch_size", "latency_avg_ms", "latency_std_ms", "fps", "ram_peak_mb"
                ])
                
                # Data rows
                for r in report.results:
                    writer.writerow([
                        r.batch_size,
                        f"{r.latency_avg:.4f}",
                        f"{r.latency_std:.4f}",
                        f"{r.fps:.4f}",
                        f"{r.ram_peak:.4f}"
                    ])
        else:
            raise ValueError(f"Format không hỗ trợ: {format}. Chọn 'csv' hoặc 'json'")
        
        print(f"✅ Benchmark report saved to: {output_path}")
        return output_path


# =============================================================================
# MAIN - EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Khởi tạo từ config file
    print("=" * 70)
    print("EXAMPLE: License Plate OCR Pipeline")
    print("=" * 70)
    
    # Đường dẫn config
    CONFIG_PATH = "pipeline_config.yaml"
    
    try:
        # Tạo pipeline từ config
        pipeline = LicensePlateOCRPipeline.from_config(CONFIG_PATH)
        
        # Load model
        pipeline.load_model()
        
        print("\n" + "=" * 70)
        print("Pipeline ready!")
        print(pipeline)
        print("=" * 70)
        
        # Example usage:
        # -----------------------------------------
        # Turn on/off preprocessing step runtime
        pipeline.enable_preprocessing_step("upscale", False)
        pipeline.enable_preprocessing_step("correct_skew", False)
        pipeline.enable_preprocessing_step("denoise", False)
        pipeline.enable_preprocessing_step("enhance_contrast", False)
        pipeline.enable_preprocessing_step("sharpen", False)
        print()
        # Single inference
        result = pipeline.run_inference("./data/cam32_20251008_081648_obj04_cls2_lp00_c080.jpg")
        print(f"Single result: {result}")
        
        # Batch inference
        results = pipeline.run_inference(["./data/cam32_20251008_081648_obj04_cls2_lp00_c080.jpg", "./data/cam32_20251008_105747_obj03_cls2_lp00_c084.jpg"])
        print(f"Batch results: {results}")
        
        # For numpy array
        image = cv2.imread("./data/cam32_20251008_105747_obj03_cls2_lp00_c084.jpg")
        result = pipeline.run_inference(image)
        print(f"Array result: {result}")
        
        # Get raw prediction (no normalize)
        raw = pipeline.get_raw_prediction("./data/cam32_20251008_105747_obj03_cls2_lp00_c084.jpg")
        print(f"Raw: {raw}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Hãy điều chỉnh đường dẫn trong pipeline_config.yaml")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()