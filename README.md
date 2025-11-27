# License Plate OCR Pipeline

Vietnamese License Plate OCR using ONNX Runtime.

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
│   ├── image_utils.py
│   └── benchmark_utils.py
├── model/                   
│   ├── model.onnx
│   └── plate_config.yaml
├── data/                     # Input images
│   └── *.jpg
└── outputs/                  # Results
```

## 🚀 Quick Start

### 1. Build Docker Image

```bash
docker build -t plate-ocr:cpu .
```

### 2. Run

```bash
docker-compose up --build
```

### 3. Interactive Mode

```bash
docker-compose run ocr bash
```

Then inside container:

```python
python
>>> from inference_pipeline import LicensePlateOCRPipeline
>>> pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")
>>> result = pipeline.run_inference("./data/plate.jpg")
>>> print(result)
```

## 📖 Usage

### Single Inference

```python
from inference_pipeline import LicensePlateOCRPipeline

pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")
result = pipeline.run_inference("./data/plate.jpg")
print(result)  # "29B112345"
```

### Batch Inference

```python
results = pipeline.run_inference([
    "./data/plate1.jpg",
    "./data/plate2.jpg",
    "./data/plate3.jpg"
])
print(results)  # ["29B112345", "51G12345", "30A67890"]
```

### Disable Preprocessing

```python
pipeline.enable_preprocessing_step("upscale", False)
pipeline.enable_preprocessing_step("denoise", False)
pipeline.enable_preprocessing_step("sharpen", False)

result = pipeline.run_inference("./data/plate.jpg")
```

### Get Raw Prediction

```python
raw = pipeline.get_raw_prediction("./data/plate.jpg")
print(raw)  # "29B1_12345__" (before normalization)
```

## 🔧 Configuration

Edit `pipeline_config.yaml`:

```yaml
model:
  onnx_model_path: "./model/model.onnx"
  plate_config_path: "./model/plate_config.yaml"
  device: "cpu"

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

postprocessing:
  normalize:
    remove_underscore: true
    remove_hyphen: true
    remove_space: true
    to_uppercase: true

inference:
  batch_size: 8
```

## 🛠 Utils

### Validate Vietnamese Plate

```python
from utils.text_utils import is_valid_plate, get_plate_type, extract_plate_parts

is_valid_plate("29B112345")  # True
is_valid_plate("ABC123")     # False

get_plate_type("29B112345")  # "motorcycle"
get_plate_type("51G12345")   # "car"
get_plate_type("TM12345")    # "military"

extract_plate_parts("51G12345")
# {"province": "51", "series": "G", "number": "12345", "type": "car"}
```

### Benchmark Speed

```python
from inference_pipeline import LicensePlateOCRPipeline
from utils.benchmark_utils import Benchmarker
import glob

pipeline = LicensePlateOCRPipeline.from_config("pipeline_config.yaml")
images = glob.glob("./data/*.jpg")

benchmarker = Benchmarker(pipeline)
report = benchmarker.run(images, batch_sizes=[1, 2, 4, 8, 16, 32])
```

Output:

```
======================================================================
BENCHMARK: Batch Size Comparison
======================================================================
Batch      Latency (ms)         FPS             RAM Peak
Size       avg ± std            img/s           (MB)
----------------------------------------------------------------------
1          36.18 ± 7.08         27.64           116.47
2          140.23 ± 49.44       14.26           129.72
4          224.84 ± 53.01       17.79           155.59
8          382.52 ± 28.46       20.91           215.34
======================================================================
📊 Best FPS: batch_size=1 (27.64 img/s)
📊 Best Latency: batch_size=1 (36.18 ms)
```

### Export Benchmark Report

```python
benchmarker.export(report, "benchmark.csv")
benchmarker.export(report, "benchmark.json", format="json")
```

## 📋 Requirements

- Python 3.10+
- OpenCV
- ONNX Runtime
- fast-plate-ocr
- PyYAML
- psutil (for benchmark)
