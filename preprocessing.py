"""
Preprocessing Module for License Plate OCR Pipeline

Chứa các hàm tiền xử lý ảnh biển số xe, có thể cấu hình on/off từng bước
thông qua file config YAML.
"""

import cv2
import numpy as np
from typing import Any, Dict, Optional, Union, List


# =============================================================================
# INDIVIDUAL PREPROCESSING FUNCTIONS
# =============================================================================

def upscale_image(
    image: np.ndarray,
    scale: float = 3,
    interpolation: str = "cubic"
) -> np.ndarray:
    """
    Phóng to ảnh theo hệ số scale.
    
    Args:
        image: Ảnh đầu vào (numpy array BGR)
        scale: Hệ số phóng to (default: 3)
        interpolation: Phương pháp nội suy ("cubic", "linear", "nearest")
        
    Returns:
        Ảnh đã được phóng to
    """
    if scale <= 0:
        raise ValueError(f"Scale phải > 0, nhận được: {scale}")
    
    if scale == 1:
        return image
    
    interp_methods = {
        "cubic": cv2.INTER_CUBIC,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST
    }
    
    interp = interp_methods.get(interpolation.lower(), cv2.INTER_CUBIC)
    
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)


def correct_skew(
    image: np.ndarray,
    blur_kernel: tuple = (5, 5),
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 20,
    max_line_gap: int = 10,
    angle_range: tuple = (-45, 45)
) -> np.ndarray:
    """
    Sửa độ nghiêng của ảnh biển số dựa trên Hough Line Transform.
    
    Args:
        image: Ảnh đầu vào (numpy array BGR)
        blur_kernel: Kernel size cho GaussianBlur
        canny_threshold1: Ngưỡng 1 cho Canny edge detection
        canny_threshold2: Ngưỡng 2 cho Canny edge detection
        hough_threshold: Ngưỡng cho HoughLinesP
        min_line_length: Độ dài tối thiểu của line
        max_line_gap: Khoảng cách tối đa giữa các điểm trên cùng line
        angle_range: Tuple (min_angle, max_angle) - chỉ xét góc trong khoảng này
        
    Returns:
        Ảnh đã được sửa nghiêng
    """
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Edge detection
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle_range[0] < angle < angle_range[1]:
                angles.append(angle)
    
    if len(angles) > 0:
        median_angle = np.median(angles)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    return image


def denoise_image(
    image: np.ndarray,
    method: str = "median",
    kernel_size: int = 3
) -> np.ndarray:
    """
    Khử nhiễu ảnh.
    
    Args:
        image: Ảnh đầu vào (numpy array BGR)
        method: Phương pháp khử nhiễu ("median", "gaussian", "bilateral")
        kernel_size: Kích thước kernel (phải là số lẻ)
        
    Returns:
        Ảnh đã được khử nhiễu
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    method = method.lower()
    
    if method == "median":
        return cv2.medianBlur(image, kernel_size)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        # Default to median
        return cv2.medianBlur(image, kernel_size)


def enhance_contrast(
    image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Tăng cường độ tương phản của ảnh.
    
    Args:
        image: Ảnh đầu vào (numpy array BGR)
        method: Phương pháp ("clahe", "histogram_eq", "none")
        clip_limit: Giới hạn clip cho CLAHE
        tile_grid_size: Kích thước tile grid cho CLAHE
        
    Returns:
        Ảnh đã được tăng cường tương phản
    """
    method = method.lower()
    
    if method == "none":
        return image
    
    if method == "histogram_eq":
        # Simple histogram equalization
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    # Default: CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_image(
    image: np.ndarray,
    kernel_type: str = "standard"
) -> np.ndarray:
    """
    Làm sắc nét ảnh.
    
    Args:
        image: Ảnh đầu vào (numpy array BGR)
        kernel_type: Loại kernel ("standard", "unsharp_mask")
        
    Returns:
        Ảnh đã được làm sắc nét
    """
    kernel_type = kernel_type.lower()
    
    if kernel_type == "unsharp_mask":
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    # Standard sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def binarize_image(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """
    Nhị phân hóa ảnh (optional - không có trong pipeline mặc định).
    
    Args:
        image: Ảnh đầu vào
        block_size: Kích thước block cho adaptive threshold
        c: Hằng số trừ đi
        
    Returns:
        Ảnh nhị phân
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c
    )
    return binary


# =============================================================================
# PREPROCESSING PIPELINE CLASS
# =============================================================================

class PreprocessingPipeline:
    """
    Class quản lý pipeline tiền xử lý với khả năng cấu hình từng bước.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo pipeline với config.
        
        Args:
            config: Dictionary chứa cấu hình preprocessing từ YAML
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate config structure"""
        required_steps = ["upscale", "correct_skew", "denoise", "enhance_contrast", "sharpen"]
        for step in required_steps:
            if step not in self.config:
                # Set default: disabled
                self.config[step] = {"enabled": False}
    
    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Đảm bảo ảnh ở format BGR 3 channels.
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh BGR 3 channels
        """
        if len(image.shape) == 2:
            # Grayscale -> BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # BGRA -> BGR
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng pipeline tiền xử lý theo thứ tự:
        upscale -> correct_skew -> denoise -> enhance_contrast -> sharpen
        
        Args:
            image: Ảnh đầu vào (numpy array)
            
        Returns:
            Ảnh đã được tiền xử lý
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Ensure BGR format
        result = self._ensure_bgr(image.copy())
        
        # 1. Upscale
        if self.config.get("upscale", {}).get("enabled", False):
            cfg = self.config["upscale"]
            result = upscale_image(
                result,
                scale=cfg.get("scale", 3),
                interpolation=cfg.get("interpolation", "cubic")
            )
        
        # 2. Correct skew
        if self.config.get("correct_skew", {}).get("enabled", False):
            cfg = self.config["correct_skew"]
            result = correct_skew(
                result,
                blur_kernel=tuple(cfg.get("blur_kernel", [5, 5])),
                canny_threshold1=cfg.get("canny_threshold1", 50),
                canny_threshold2=cfg.get("canny_threshold2", 150),
                hough_threshold=cfg.get("hough_threshold", 80),
                min_line_length=cfg.get("min_line_length", 20),
                max_line_gap=cfg.get("max_line_gap", 10),
                angle_range=tuple(cfg.get("angle_range", [-45, 45]))
            )
        
        # 3. Denoise
        if self.config.get("denoise", {}).get("enabled", False):
            cfg = self.config["denoise"]
            result = denoise_image(
                result,
                method=cfg.get("method", "median"),
                kernel_size=cfg.get("kernel_size", 3)
            )
        
        # 4. Enhance contrast
        if self.config.get("enhance_contrast", {}).get("enabled", False):
            cfg = self.config["enhance_contrast"]
            result = enhance_contrast(
                result,
                method=cfg.get("method", "clahe"),
                clip_limit=cfg.get("clip_limit", 2.0),
                tile_grid_size=tuple(cfg.get("tile_grid_size", [8, 8]))
            )
        
        # 5. Sharpen
        if self.config.get("sharpen", {}).get("enabled", False):
            cfg = self.config["sharpen"]
            result = sharpen_image(
                result,
                kernel_type=cfg.get("kernel_type", "standard")
            )
        
        return result
    
    def get_enabled_steps(self) -> List[str]:
        """Trả về danh sách các bước được bật"""
        steps = ["upscale", "correct_skew", "denoise", "enhance_contrast", "sharpen"]
        return [s for s in steps if self.config.get(s, {}).get("enabled", False)]
    
    def __repr__(self) -> str:
        enabled = self.get_enabled_steps()
        return f"PreprocessingPipeline(enabled_steps={enabled})"