# 🔍 Visionary Extractor
### Amazon ML Challenge 2024 — Feature Extraction from Product Images

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Challenge-Amazon%20ML%202024-orange?logo=amazon&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Image%20Entity%20Extraction-purple"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
</p>

---

## 📌 Overview

**Visionary Extractor** is a machine learning solution developed for the **Amazon ML Challenge 2024**. The goal is to build a model that extracts entity values (such as weight, volume, voltage, dimensions, etc.) directly from product images — a critical capability for e-commerce platforms where textual descriptions are often missing or incomplete.

> 💡 Digital marketplaces often carry products without detailed textual metadata. This solution bridges that gap by extracting key product attributes directly from images.

---

## 🎯 Problem Statement

Given a product image and an entity name (e.g., `item_weight`, `voltage`, `height`), the model must predict the entity value in a valid format:

```
"<float> <unit>"   →   e.g., "34 gram", "12.5 centimetre", "110 volt"
```

Predictions are evaluated using the **F1 Score** across all entity types.

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `dataset/train.csv` | Training data with ground truth `entity_value` |
| `dataset/test.csv` | Test data — predict `entity_value` for each row |
| `dataset/sample_test.csv` | Sample test input |
| `dataset/sample_test_out.csv` | Sample output format reference |

### Dataset Columns

| Column | Description |
|--------|-------------|
| `index` | Unique identifier for each sample |
| `image_link` | Public URL of the product image |
| `group_id` | Product category code |
| `entity_name` | Entity to extract (e.g., `item_weight`, `voltage`) |
| `entity_value` | *(Target)* Value in `"x unit"` format — only in train set |

---

## 🏗️ Project Structure

```
visionary_extractor/
│
├── cognition_24/                  # Core ML package
│   ├── __init__.py
│   ├── core/                      # Extraction logic & model inference
│   ├── models/                    # ML/Vision model definitions
│   ├── processors/                # Pre & post-processing pipelines
│   └── utils/                     # Helper functions, logging
│
├── src/
│   ├── constants.py               # Allowed units per entity type
│   ├── utils.py                   # Image downloader & helpers
│   ├── sanity.py                  # Output format validator
│   └── test.ipynb                 # Sample notebook & usage demo
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_test.csv
│   └── sample_test_out.csv
│
├── sample_code.py                 # Dummy output generator (reference)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Stable internet connection (for downloading product images)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Dipak10016/visionary_extractor.git
cd visionary_extractor

# 2. Create and activate a virtual environment
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🖼️ Downloading Images

Use the provided utility to download product images from their URLs:

```python
from src.utils import download_images

# Download images from the dataset
download_images(
    image_links=df['image_link'].tolist(),
    download_folder='images/'
)
```

---

## 🔍 Running Extraction

```python
from cognition_24.core import VisionaryExtractor

# Initialize extractor
extractor = VisionaryExtractor()

# Extract entity value from an image
result = extractor.extract(
    image_path="images/product_001.jpg",
    entity_name="item_weight"
)

print(result)  # Output: "34 gram"
```

### Batch Inference on Test Set

```python
import pandas as pd
from cognition_24.core import VisionaryExtractor

df_test = pd.read_csv("dataset/test.csv")
extractor = VisionaryExtractor()

predictions = []
for _, row in df_test.iterrows():
    pred = extractor.extract(
        image_link=row['image_link'],
        entity_name=row['entity_name']
    )
    predictions.append({"index": row['index'], "prediction": pred})

output_df = pd.DataFrame(predictions)
output_df.to_csv("test_out.csv", index=False)
```

---

## ✅ Output Format

All predictions must follow this exact format:

```
"<float> <unit>"
```

| ✅ Valid | ❌ Invalid |
|---------|-----------|
| `"2 gram"` | `"2 gms"` |
| `"12.5 centimetre"` | `"60 ounce/1.7 kilogram"` |
| `"2.56 ounce"` | `"2.2e2 kilogram"` |
| `""` *(if no value found)* | Any unlisted unit |

> ⚠️ Make sure predictions are output for **all indices**. Return `""` if no value is found.

---

## 🧪 Sanity Check

Before submitting, validate your output file:

```bash
python src/sanity.py --file test_out.csv
```

✅ A successful check will print:
```
Parsing successful for file: test_out.csv
```

> ⚠️ Files that fail the sanity check will **not be evaluated** on the leaderboard.

---

## 📏 Allowed Entity Units

<details>
<summary>Click to expand the full unit map</summary>

| Entity | Allowed Units |
|--------|--------------|
| `width`, `depth`, `height` | centimetre, foot, millimetre, metre, inch, yard |
| `item_weight` | milligram, kilogram, microgram, gram, ounce, ton, pound |
| `maximum_weight_recommendation` | milligram, kilogram, microgram, gram, ounce, ton, pound |
| `voltage` | millivolt, kilovolt, volt |
| `wattage` | kilowatt, watt |
| `item_volume` | cubic foot, microlitre, cup, fluid ounce, centilitre, imperial gallon, pint, decilitre, litre, millilitre, quart, cubic inch, gallon |

</details>

Full mapping is available in [`src/constants.py`](src/constants.py).

---

## 📊 Evaluation Metric — F1 Score

| Case | Condition |
|------|-----------|
| ✅ True Positive | `OUT != ""` and `GT != ""` and `OUT == GT` |
| ❌ False Positive | `OUT != ""` and `GT != ""` and `OUT != GT` |
| ❌ False Positive | `OUT != ""` and `GT == ""` |
| ❌ False Negative | `OUT == ""` and `GT != ""` |
| ➖ True Negative | `OUT == ""` and `GT == ""` |

```
F1 = 2 × Precision × Recall / (Precision + Recall)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Image Processing | OpenCV, Pillow |
| ML / Vision | Transformers, scikit-learn |
| Data Handling | Pandas, NumPy |
| OCR | Tesseract / Vision APIs |
| Data Validation | Pydantic |
| Testing | Pytest |

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and write tests
4. Commit: `git commit -m "Add: your feature description"`
5. Push: `git push origin feature/your-feature`
6. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Dipak**  
GitHub: [@Dipak10016](https://github.com/Dipak10016)

---

<p align="center">⭐ Star this repo if you found it helpful!</p>
<p align="center">Made with ❤️ for Amazon ML Challenge 2024</p>
