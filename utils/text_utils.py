"""
Text Utilities for Vietnamese License Plate OCR

Các hàm xử lý và validate text biển số xe Việt Nam.

Quy tắc biển số Việt Nam:
- Biển xe máy: XX-Y1 XXX.XX hoặc XX-Y1 XXXX (VD: 29-B1 123.45, 77L1-12345)
- Biển ô tô: XX-Y XXXX hoặc XX-Y XXXXX (VD: 30A-12345, 51G-123.45)
- XX: Mã tỉnh/thành phố (11-99)
- Y: Seri (A-Z, không có I, O, Q, W)
- Số: 4-5 chữ số

Biển số đặc biệt:
- Biển xanh (công an): 80A, 80B...
- Biển đỏ (quân đội): Bắt đầu bằng chữ (VD: TM, KA, KB...)
- Biển ngoại giao: NG, CV, QT...
"""

import re
from typing import Optional, Tuple, Dict

# =============================================================================
# CONSTANTS - Quy tắc biển số Việt Nam
# =============================================================================

# Mã tỉnh/thành phố hợp lệ (11-99, một số mã đặc biệt)
VALID_PROVINCE_CODES = set([
    11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99,
    80,  # Công an, quân đội
])

# Seri hợp lệ (không có I, O, Q, W vì dễ nhầm với số)
VALID_SERIES = set("ABCDEFGHKLMNPRSTUVXYZ")

# Seri đặc biệt cho xe máy (có số đằng sau)
MOTORCYCLE_SERIES = set("ABCDEFGHKLMNPRSTUVXYZ")

# Prefix biển đỏ (quân đội)
MILITARY_PREFIXES = [
    "TM", "TH", "TT", "TK", "TC", "TB", "TN", "TP",
    "KA", "KB", "KC", "KD", "KK", "KV", "KT",
    "PA", "PB", "PC", "PD", "PE", "PK",
    "QA", "QB", "QC", "QD", "QK", "QH",
    "AA", "AB", "AC", "AD", "AH", "AK", "AM", "AN", "AP", "AT", "AV",
    "BA", "BB", "BC", "BD", "BK", "BL", "BM", "BT",
    "CA", "CB", "CC", "CD", "CH", "CK", "CN", "CP",
    "HA", "HB", "HC", "HD", "HE", "HH", "HK", "HN", "HP", "HQ", "HT",
    "VT", "VB", "VK",
]

# Prefix biển ngoại giao
DIPLOMATIC_PREFIXES = ["NG", "CV", "QT"]


# =============================================================================
# NORMALIZE FUNCTIONS
# =============================================================================

def normalize_plate(plate_text: str) -> str:
    """
    Chuẩn hóa text biển số.
    
    Xử lý:
    - Xóa ký tự padding (_)
    - Xóa dấu gạch ngang (-)
    - Xóa dấu chấm (.)
    - Xóa khoảng trắng
    - Chuyển thành chữ hoa
    
    Args:
        plate_text: Text biển số thô
        
    Returns:
        Text đã chuẩn hóa
        
    Examples:
        >>> normalize_plate("29-B1 123.45")
        "29B112345"
        >>> normalize_plate("51G_12345__")
        "51G12345"
        >>> normalize_plate("30a-123.45")
        "30A12345"
    """
    if not plate_text:
        return ""
    
    result = plate_text.upper()
    
    # Xóa các ký tự không mong muốn
    chars_to_remove = ["_", "-", ".", " ", "\t", "\n"]
    for char in chars_to_remove:
        result = result.replace(char, "")
    
    return result


def normalize_for_comparison(plate1: str, plate2: str) -> Tuple[str, str]:
    """
    Chuẩn hóa 2 biển số để so sánh.
    
    Args:
        plate1: Biển số thứ nhất
        plate2: Biển số thứ hai
        
    Returns:
        Tuple (normalized_plate1, normalized_plate2)
    """
    return normalize_plate(plate1), normalize_plate(plate2)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_valid_plate(plate_text: str, strict: bool = False) -> bool:
    """
    Kiểm tra biển số có hợp lệ theo quy tắc Việt Nam không.
    
    Args:
        plate_text: Text biển số (đã normalize hoặc chưa)
        strict: True = kiểm tra chặt (cả mã tỉnh), False = kiểm tra cơ bản
        
    Returns:
        True nếu hợp lệ, False nếu không
        
    Examples:
        >>> is_valid_plate("29B112345")
        True
        >>> is_valid_plate("51G12345")
        True
        >>> is_valid_plate("ABC123")
        False
        >>> is_valid_plate("80A12345")  # Biển công an
        True
    """
    if not plate_text:
        return False
    
    # Normalize trước
    normalized = normalize_plate(plate_text)
    
    if len(normalized) < 7 or len(normalized) > 10:
        return False
    
    # Pattern 1: Biển thường - XX + Seri + Số (VD: 29B112345, 51G12345)
    # Pattern ô tô: 2 số + 1 chữ + 4-5 số
    pattern_car = r"^(\d{2})([A-Z])(\d{4,5})$"
    
    # Pattern xe máy: 2 số + 1 chữ + 1 số + 4-5 số (VD: 29B112345)
    pattern_motorcycle = r"^(\d{2})([A-Z])(\d{1})(\d{4,5})$"
    
    # Pattern biển quân đội: 2 chữ + số
    pattern_military = r"^([A-Z]{2})(\d{4,6})$"
    
    # Pattern biển ngoại giao: NG/CV/QT + số
    pattern_diplomatic = r"^(NG|CV|QT)(\d{4,6})$"
    
    # Thử match các pattern
    match_car = re.match(pattern_car, normalized)
    match_motorcycle = re.match(pattern_motorcycle, normalized)
    match_military = re.match(pattern_military, normalized)
    match_diplomatic = re.match(pattern_diplomatic, normalized)
    
    if match_car:
        province_code = int(match_car.group(1))
        series = match_car.group(2)
        
        if strict:
            if province_code not in VALID_PROVINCE_CODES:
                return False
            if series not in VALID_SERIES:
                return False
        else:
            # Basic check
            if province_code < 11 or province_code > 99:
                return False
        
        return True
    
    if match_motorcycle:
        province_code = int(match_motorcycle.group(1))
        series = match_motorcycle.group(2)
        
        if strict:
            if province_code not in VALID_PROVINCE_CODES:
                return False
            if series not in VALID_SERIES:
                return False
        else:
            if province_code < 11 or province_code > 99:
                return False
        
        return True
    
    if match_military:
        prefix = match_military.group(1)
        if strict:
            return prefix in MILITARY_PREFIXES
        return True
    
    if match_diplomatic:
        return True
    
    return False


def get_plate_type(plate_text: str) -> str:
    """
    Xác định loại biển số.
    
    Args:
        plate_text: Text biển số
        
    Returns:
        Loại biển: "car", "motorcycle", "military", "diplomatic", "unknown"
        
    Examples:
        >>> get_plate_type("51G12345")
        "car"
        >>> get_plate_type("29B112345")
        "motorcycle"
        >>> get_plate_type("TM12345")
        "military"
    """
    if not plate_text:
        return "unknown"
    
    normalized = normalize_plate(plate_text)
    
    # Check diplomatic
    if re.match(r"^(NG|CV|QT)\d+$", normalized):
        return "diplomatic"
    
    # Check military
    if re.match(r"^[A-Z]{2}\d+$", normalized):
        prefix = normalized[:2]
        if prefix in MILITARY_PREFIXES:
            return "military"
    
    # Check motorcycle (có số ở vị trí thứ 4)
    if re.match(r"^\d{2}[A-Z]\d{1}\d{4,5}$", normalized):
        return "motorcycle"
    
    # Check car
    if re.match(r"^\d{2}[A-Z]\d{4,5}$", normalized):
        return "car"
    
    return "unknown"


# =============================================================================
# EXTRACT FUNCTIONS
# =============================================================================

def extract_plate_parts(plate_text: str) -> Optional[Dict[str, str]]:
    """
    Tách biển số thành các phần.
    
    Args:
        plate_text: Text biển số
        
    Returns:
        Dict với các phần hoặc None nếu không parse được
        
    Examples:
        >>> extract_plate_parts("29B112345")
        {"province": "29", "series": "B1", "number": "12345", "type": "motorcycle"}
        
        >>> extract_plate_parts("51G12345")
        {"province": "51", "series": "G", "number": "12345", "type": "car"}
        
        >>> extract_plate_parts("TM12345")
        {"prefix": "TM", "number": "12345", "type": "military"}
    """
    if not plate_text:
        return None
    
    normalized = normalize_plate(plate_text)
    plate_type = get_plate_type(normalized)
    
    if plate_type == "unknown":
        return None
    
    result = {"type": plate_type, "raw": plate_text, "normalized": normalized}
    
    if plate_type == "car":
        match = re.match(r"^(\d{2})([A-Z])(\d{4,5})$", normalized)
        if match:
            result["province"] = match.group(1)
            result["series"] = match.group(2)
            result["number"] = match.group(3)
    
    elif plate_type == "motorcycle":
        match = re.match(r"^(\d{2})([A-Z])(\d{1})(\d{4,5})$", normalized)
        if match:
            result["province"] = match.group(1)
            result["series"] = match.group(2) + match.group(3)
            result["number"] = match.group(4)
    
    elif plate_type == "military":
        match = re.match(r"^([A-Z]{2})(\d+)$", normalized)
        if match:
            result["prefix"] = match.group(1)
            result["number"] = match.group(2)
    
    elif plate_type == "diplomatic":
        match = re.match(r"^(NG|CV|QT)(\d+)$", normalized)
        if match:
            result["prefix"] = match.group(1)
            result["number"] = match.group(2)
    
    return result


def get_province_name(province_code: str) -> str:
    """
    Lấy tên tỉnh/thành phố từ mã.
    
    Args:
        province_code: Mã tỉnh (2 số)
        
    Returns:
        Tên tỉnh/thành phố hoặc "Unknown"
    """
    PROVINCE_NAMES = {
        "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng",
        "16": "Hải Phòng", "17": "Thái Bình", "18": "Nam Định", "19": "Phú Thọ",
        "20": "Thái Nguyên", "21": "Yên Bái", "22": "Tuyên Quang", "23": "Hà Giang",
        "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La", "27": "Điện Biên",
        "28": "Hòa Bình", "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội",
        "32": "Hà Nội", "33": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình",
        "36": "Thanh Hóa", "37": "Nghệ An", "38": "Hà Tĩnh", "39": "Đồng Nai",
        "40": "Hà Nội", "41": "Hà Nội",
        "43": "Đà Nẵng", "47": "Đắk Lắk", "48": "Đắk Nông",
        "49": "Lâm Đồng", "50": "TP.HCM", "51": "TP.HCM", "52": "TP.HCM",
        "53": "TP.HCM", "54": "TP.HCM", "55": "TP.HCM", "56": "TP.HCM",
        "57": "TP.HCM", "58": "TP.HCM", "59": "TP.HCM",
        "60": "Đồng Nai", "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang",
        "64": "Vĩnh Long", "65": "Cần Thơ", "66": "Đồng Tháp", "67": "An Giang",
        "68": "Kiên Giang", "69": "Cà Mau", "70": "Tây Ninh", "71": "Bến Tre",
        "72": "Bà Rịa-Vũng Tàu", "73": "Quảng Bình", "74": "Quảng Trị",
        "75": "Thừa Thiên Huế", "76": "Quảng Ngãi", "77": "Bình Định",
        "78": "Phú Yên", "79": "Khánh Hòa",
        "81": "Gia Lai", "82": "Kon Tum", "83": "Sóc Trăng", "84": "Trà Vinh",
        "85": "Ninh Thuận", "86": "Bình Thuận", "88": "Vĩnh Phúc",
        "89": "Hưng Yên", "90": "Hà Nam", "92": "Quảng Nam", "93": "Bình Phước",
        "94": "Bạc Liêu", "95": "Hậu Giang", "97": "Bắc Kạn", "99": "Bắc Ninh",
        "80": "Cơ quan TW / Quân đội",
    }
    
    return PROVINCE_NAMES.get(province_code, "Unknown")


# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Tính khoảng cách Levenshtein (edit distance) giữa 2 chuỗi.
    
    Args:
        s1: Chuỗi thứ nhất
        s2: Chuỗi thứ hai
        
    Returns:
        Số bước chỉnh sửa tối thiểu
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def plate_similarity(plate1: str, plate2: str) -> float:
    """
    Tính độ tương đồng giữa 2 biển số (0.0 - 1.0).
    
    Args:
        plate1: Biển số thứ nhất
        plate2: Biển số thứ hai
        
    Returns:
        Độ tương đồng (1.0 = giống hoàn toàn)
        
    Examples:
        >>> plate_similarity("29B112345", "29B112345")
        1.0
        >>> plate_similarity("29B112345", "29B112346")
        0.888...
    """
    norm1, norm2 = normalize_for_comparison(plate1, plate2)
    
    if not norm1 or not norm2:
        return 0.0
    
    if norm1 == norm2:
        return 1.0
    
    distance = levenshtein_distance(norm1, norm2)
    max_len = max(len(norm1), len(norm2))
    
    return 1.0 - (distance / max_len)


def plates_match(plate1: str, plate2: str) -> bool:
    """
    Kiểm tra 2 biển số có khớp nhau không (sau khi normalize).
    
    Args:
        plate1: Biển số thứ nhất
        plate2: Biển số thứ hai
        
    Returns:
        True nếu khớp, False nếu không
    """
    norm1, norm2 = normalize_for_comparison(plate1, plate2)
    return norm1 == norm2 and len(norm1) > 0
