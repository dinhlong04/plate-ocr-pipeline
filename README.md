# License Plate OCR Pipeline

Vietnamese License Plate OCR using ONNX Runtime (CPU).

## 📁 Project Structure

```
plate-ocr-pipeline/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pipeline_config.yaml
├── inference_pipeline.py
├── preprocessing.py
├── utils/
│   ├── __init__.py
│   ├── text_utils.py
│   └── image_utils.py
├── model/                   
│   ├── model.onnx
│   └── plate_config.yaml
├── data/                     # Input images
│   └── *.jpg
├── label/                     # Truth label
│   └── *.csv
└── outputs/                  # Results
```

## 🚀 Quick Start

### 1. Build Docker image

```bash
docker build -t plate-ocr:cpu .
```

### 2. Run

**Docker Compose**

```bash
docker-compose up --build
```

**If you want to try another image**

```bash
docker-compose run ocr bash
```
Then inside container:

```python
from inference_pipeline import LicensePlateOCRPipeline

pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")
result = pipeline.run_inference("./data/cam1_20250926_142949_car_1_plate_0_0.jpg")
print(result)
```

## 📖 Usage Examples

### Single Inference

```python
from inference_pipeline import LicensePlateOCRPipeline

pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")

# Single image
result = pipeline.run_inference("./data/cam16_20251009_093709_obj01_cls1_lp00_c085.jpg")
print(f"Result: {result}")
```

### Batch Inference

```python
# Multiple images
results = pipeline.run_inference([
    "./data/cam16_20251009_093709_obj01_cls1_lp00_c085.jpg",
    "./data/cam32_20251008_081648_obj04_cls2_lp00_c080.jpg",
    "./data/cam2_20250926_134219_obj06_cls1_lp00_c084.jpg"
])
print(f"Results: {results}")
```

### Disable Preprocessing Steps

```python
# Disable upscale and sharpen
pipeline.enable_preprocessing_step("upscale", False)
pipeline.enable_preprocessing_step("sharpen", False)

result = pipeline.run_inference("data/cam16_20251009_093709_obj01_cls1_lp00_c085.jpg")
print(result)
```

### Get Raw Prediction (No Postprocess)

```python
raw = pipeline.get_raw_prediction("data/cam16_20251009_093709_obj01_cls1_lp00_c085.jpg")
print(f"Raw: {raw}")  # May contain underscores: "29B1_12345__"
```

## 🔧 Configuration

Edit `pipeline_config.yaml` to customize:

### Preprocessing Steps

```yaml
preprocessing:
  upscale:
    enabled: true
    scale: 3
  
  correct_skew:
    enabled: true
  
  denoise:
    enabled: true
    method: "median"
  
  enhance_contrast:
    enabled: true
    method: "clahe"
  
  sharpen:
    enabled: true
```

### Postprocessing

```yaml
postprocessing:
  normalize:
    remove_underscore: true
    remove_hyphen: true
    to_uppercase: true
```

## 🛠 Utils

### Validate Vietnamese Plate

```python
from utils.text_utils import is_valid_plate, get_plate_type, extract_plate_parts

# Validate
is_valid_plate("29B112345")  # True
is_valid_plate("ABC123")     # False

# Get type
get_plate_type("29B112345")  # "motorcycle"
get_plate_type("51G12345")   # "car"
get_plate_type("TM12345")    # "military"

# Extract parts
extract_plate_parts("29B112345")
# {"province": "29", "series": "B1", "number": "12345", "type": "motorcycle"}
```

**BENCHMARK SPEED**

```python

# Benchmark batch sizes (1 → 32)
images = ["./data/cam1_20250926_132743_car_0_plate_0_0.jpg", "./data/cam1_20250926_133824_car_2_plate_0_0.jpg", 
          "./data/cam1_20250926_142602_car_3_plate_0_0.jpg", "./data/cam1_20250926_142804_car_0_plate_0_0.jpg",
          "./data/cam1_20250926_142949_car_1_plate_0_0.jpg", "./data/cam1_20250926_143554_car_0_plate_0_0.jpg",
          "./data/cam1_20251008_105152_car_0_plate_0_0.jpg", "./data/cam2_20250926_133219_obj03_cls1_lp00_c084.jpg",
          "./data/cam3_20251008_081400_obj04_cls2_lp00_c078.jpg", "./data/cam3_20251008_081437_obj06_cls1_lp00_c084.jpg",
          "./data/cam3_20251008_081456_obj00_cls1_lp00_c079.jpg", "./data/cam3_20251008_104014_obj00_cls1_lp00_c084.jpg",
          "./data/cam3_20251008_104014_obj02_cls1_lp00_c082.jpg", "./data/cam3_20251008_104014_obj03_cls1_lp00_c081.jpg",
          "./data/cam3_20251008_104136_obj03_cls1_lp00_c081.jpg", "./data/cam3_20251008_104339_obj00_cls1_lp00_c084.jpg",
          "./data/cam3_20251008_104517_obj00_cls1_lp00_c071.jpg", "./data/cam3_20251008_104555_obj03_cls1_lp00_c080.jpg",
          "./data/cam3_20251008_104555_obj04_cls1_lp00_c081.jpg", "./data/cam3_20251008_105050_obj03_cls1_lp00_c080.jpg", 
          "./data/cam3_20251008_105050_obj04_cls1_lp00_c082.jpg", "./data/cam3_20251008_105050_obj05_cls1_lp00_c082.jpg", 
          "./data/cam3_20251008_105050_obj07_cls1_lp00_c078.jpg", "./data/cam3_20251008_105233_obj00_cls1_lp00_c082.jpg", 
          "./data/cam3_20251008_105233_obj01_cls1_lp00_c085.jpg", "./data/cam3_20251008_105253_obj00_cls1_lp00_c084.jpg",
          "./data/cam3_20251008_105431_obj03_cls1_lp00_c082.jpg", "./data/cam3_20251008_105905_obj03_cls1_lp00_c083.jpg",
          "./data/cam3_20251008_110022_obj04_cls2_lp00_c083.jpg", "./data/cam3_20251008_110256_obj00_cls1_lp00_c080.jpg",
          "./data/cam3_20251008_110256_obj03_cls2_lp00_c080.jpg", "./data/cam3_20251008_110315_obj06_cls1_lp00_c083.jpg"]  # >= 32 ảnh
report = pipeline.benchmark_batch_sizes(
    images, 
    batch_sizes=[1, 2, 4, 8, 16, 32],
    n_runs=50
)

# Export
pipeline.export_benchmark_report(report, "benchmark.csv", format="csv")

```
