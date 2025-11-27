"""
Image Utilities for License Plate OCR Pipeline

Các hàm tiện ích xử lý ảnh đầu vào.
"""

import os
import base64
import numpy as np
import cv2
from typing import Union, Optional, Tuple


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(
    source: Union[str, bytes, np.ndarray],
    flags: int = cv2.IMREAD_COLOR
) -> Optional[np.ndarray]:
    """
    Load ảnh từ nhiều nguồn khác nhau.
    
    Hỗ trợ:
    - Đường dẫn file (str)
    - Bytes (raw bytes hoặc base64)
    - Numpy array (trả về trực tiếp)
    
    Args:
        source: Nguồn ảnh (path, bytes, hoặc numpy array)
        flags: OpenCV imread flags (default: IMREAD_COLOR = BGR)
        
    Returns:
        numpy array (BGR) hoặc None nếu lỗi
        
    Examples:
        >>> img = load_image("plate.jpg")
        >>> img = load_image(image_bytes)
        >>> img = load_image(numpy_array)
    """
    # Nếu đã là numpy array
    if isinstance(source, np.ndarray):
        return source.copy()
    
    # Nếu là đường dẫn file
    if isinstance(source, str):
        # Check if base64 string
        if source.startswith("data:image"):
            # Data URL format: data:image/jpeg;base64,/9j/4AAQ...
            try:
                header, encoded = source.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(image_bytes, np.uint8)
                return cv2.imdecode(nparr, flags)
            except Exception:
                return None
        
        # Check if pure base64 string (no header)
        if len(source) > 100 and not os.path.exists(source):
            try:
                image_bytes = base64.b64decode(source)
                nparr = np.frombuffer(image_bytes, np.uint8)
                return cv2.imdecode(nparr, flags)
            except Exception:
                pass  # Not base64, try as file path
        
        # Regular file path
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        
        image = cv2.imread(source, flags)
        if image is None:
            raise ValueError(f"Cannot read image: {source}")
        return image
    
    # Nếu là bytes
    if isinstance(source, bytes):
        nparr = np.frombuffer(source, np.uint8)
        image = cv2.imdecode(nparr, flags)
        if image is None:
            raise ValueError("Cannot decode image from bytes")
        return image
    
    raise TypeError(f"Unsupported source type: {type(source)}")


def load_image_safe(
    source: Union[str, bytes, np.ndarray],
    flags: int = cv2.IMREAD_COLOR
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load ảnh với error handling (không raise exception).
    
    Args:
        source: Nguồn ảnh
        flags: OpenCV imread flags
        
    Returns:
        Tuple (image, error_message)
        - Nếu thành công: (numpy_array, None)
        - Nếu lỗi: (None, error_string)
    """
    try:
        image = load_image(source, flags)
        return image, None
    except FileNotFoundError as e:
        return None, f"File not found: {e}"
    except ValueError as e:
        return None, f"Invalid image: {e}"
    except TypeError as e:
        return None, f"Type error: {e}"
    except Exception as e:
        return None, f"Unknown error: {e}"


# =============================================================================
# IMAGE FORMAT CONVERSION
# =============================================================================

def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Đảm bảo ảnh ở format BGR 3 channels.
    
    Chuyển đổi:
    - Grayscale (1 channel) -> BGR
    - BGRA (4 channels) -> BGR
    - RGB -> BGR (nếu cần)
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh BGR 3 channels
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # Already BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    
    # Grayscale -> BGR
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # BGRA -> BGR
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    raise ValueError(f"Unsupported image shape: {image.shape}")


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Đảm bảo ảnh ở format grayscale 1 channel.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh grayscale
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # Already grayscale
    if len(image.shape) == 2:
        return image
    
    # BGR -> Grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # BGRA -> Grayscale
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    raise ValueError(f"Unsupported image shape: {image.shape}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Chuyển BGR sang RGB (cho display với matplotlib, PIL, etc.)
    
    Args:
        image: Ảnh BGR
        
    Returns:
        Ảnh RGB
    """
    if len(image.shape) == 2:
        return image  # Grayscale, no conversion needed
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Chuyển RGB sang BGR (cho OpenCV processing)
    
    Args:
        image: Ảnh RGB
        
    Returns:
        Ảnh BGR
    """
    if len(image.shape) == 2:
        return image  # Grayscale, no conversion needed
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# =============================================================================
# IMAGE RESIZING
# =============================================================================

def resize_to_model_input(
    image: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_CUBIC,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize ảnh về kích thước model yêu cầu.
    
    Args:
        image: Ảnh đầu vào
        target_width: Chiều rộng mong muốn (None = giữ nguyên tỷ lệ theo height)
        target_height: Chiều cao mong muốn (None = giữ nguyên tỷ lệ theo width)
        keep_aspect_ratio: Giữ tỷ lệ khung hình (padding nếu cần)
        interpolation: Phương pháp nội suy
        pad_color: Màu padding (BGR)
        
    Returns:
        Ảnh đã resize
    """
    if image is None:
        raise ValueError("Input image is None")
    
    h, w = image.shape[:2]
    
    # Nếu không chỉ định kích thước, trả về nguyên bản
    if target_width is None and target_height is None:
        return image
    
    # Tính kích thước mới
    if target_width is None:
        # Chỉ có target_height
        scale = target_height / h
        new_w = int(w * scale)
        new_h = target_height
    elif target_height is None:
        # Chỉ có target_width
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
    else:
        # Có cả hai
        if keep_aspect_ratio:
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w = target_width
            new_h = target_height
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Padding nếu cần
    if keep_aspect_ratio and target_width is not None and target_height is not None:
        if new_w != target_width or new_h != target_height:
            # Tạo canvas với màu padding
            if len(image.shape) == 3:
                canvas = np.full((target_height, target_width, image.shape[2]), pad_color, dtype=np.uint8)
            else:
                canvas = np.full((target_height, target_width), pad_color[0], dtype=np.uint8)
            
            # Đặt ảnh vào giữa
            x_offset = (target_width - new_w) // 2
            y_offset = (target_height - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
    
    return resized


def resize_by_scale(
    image: np.ndarray,
    scale: float,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Resize ảnh theo hệ số scale.
    
    Args:
        image: Ảnh đầu vào
        scale: Hệ số scale (VD: 2.0 = phóng to gấp đôi)
        interpolation: Phương pháp nội suy
        
    Returns:
        Ảnh đã resize
    """
    if scale <= 0:
        raise ValueError(f"Scale must be > 0, got: {scale}")
    
    if scale == 1.0:
        return image
    
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


# =============================================================================
# IMAGE VALIDATION
# =============================================================================

def is_valid_image(image: np.ndarray) -> bool:
    """
    Kiểm tra ảnh có hợp lệ không.
    
    Args:
        image: Ảnh cần kiểm tra
        
    Returns:
        True nếu hợp lệ
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if image.size == 0:
        return False
    
    # Phải có ít nhất 2 chiều (h, w)
    if len(image.shape) < 2:
        return False
    
    # Nếu có 3 chiều, channel phải là 1, 3, hoặc 4
    if len(image.shape) == 3:
        if image.shape[2] not in [1, 3, 4]:
            return False
    
    return True


def get_image_info(image: np.ndarray) -> dict:
    """
    Lấy thông tin về ảnh.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Dict với thông tin ảnh
    """
    if not is_valid_image(image):
        return {"valid": False}
    
    info = {
        "valid": True,
        "height": image.shape[0],
        "width": image.shape[1],
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes,
        "size_mb": image.nbytes / (1024 * 1024),
    }
    
    # Xác định format
    if len(image.shape) == 2:
        info["format"] = "grayscale"
    elif image.shape[2] == 3:
        info["format"] = "BGR"
    elif image.shape[2] == 4:
        info["format"] = "BGRA"
    else:
        info["format"] = "unknown"
    
    return info


# =============================================================================
# IMAGE ENCODING
# =============================================================================

def image_to_bytes(
    image: np.ndarray,
    format: str = ".jpg",
    quality: int = 95
) -> bytes:
    """
    Chuyển ảnh thành bytes.
    
    Args:
        image: Ảnh đầu vào
        format: Format (".jpg", ".png", ".webp")
        quality: Chất lượng (0-100, chỉ cho JPEG/WebP)
        
    Returns:
        Bytes của ảnh
    """
    encode_params = []
    
    if format.lower() in [".jpg", ".jpeg"]:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format.lower() == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - int(quality / 11)]
    elif format.lower() == ".webp":
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    
    success, encoded = cv2.imencode(format, image, encode_params)
    
    if not success:
        raise ValueError(f"Failed to encode image to {format}")
    
    return encoded.tobytes()


def image_to_base64(
    image: np.ndarray,
    format: str = ".jpg",
    quality: int = 95,
    with_header: bool = True
) -> str:
    """
    Chuyển ảnh thành base64 string.
    
    Args:
        image: Ảnh đầu vào
        format: Format (".jpg", ".png")
        quality: Chất lượng
        with_header: Thêm data URL header
        
    Returns:
        Base64 string
    """
    image_bytes = image_to_bytes(image, format, quality)
    b64_string = base64.b64encode(image_bytes).decode("utf-8")
    
    if with_header:
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        mime = mime_types.get(format.lower(), "image/jpeg")
        return f"data:{mime};base64,{b64_string}"
    
    return b64_string
