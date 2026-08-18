# Layout-Aware Document Parser

<p align="center">
  <img src="https://img.shields.io/badge/NLP-LayoutLMv3-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Dataset-FUNSD-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/OCR-PyMuPDF-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Task-Document%20Understanding-purple?style=for-the-badge">
</p>

<p align="center">
A layout-aware document understanding pipeline using LayoutLMv3 to extract structured information from form-based documents while combining textual, visual, and spatial features.
</p>

---

## 📌 Overview

This project fine-tunes **LayoutLMv3** for token classification using the **FUNSD** dataset and applies the trained model to PDF documents using **PyMuPDF**.

The pipeline combines document text, images, and **2D spatial information** to identify entities such as **QUESTION, ANSWER, HEADER, and O**, then converts token-level predictions into structured document regions.

---

## ✨ Features

* Fine-tuning **LayoutLMv3** on FUNSD
* Layout-aware token classification using text and bounding boxes
* Custom PyTorch dataset for word, box, and label alignment
* Baseline vs fine-tuned performance comparison
* PDF word and bounding-box extraction using **PyMuPDF**
* BIO-based entity prediction and region grouping
* Confidence-aware structured JSON output
* Visual comparison of predictions and ground truth
* End-to-end training and inference in a single notebook

---

## 📐 Evaluation Metrics

| Metric          | Purpose                                  |
| --------------- | ---------------------------------------- |
| Precision       | Measures correct entity predictions      |
| Recall          | Measures entity coverage                 |
| F1 Score        | Overall token-classification performance |
| Confidence      | Measures prediction certainty            |
| Region Accuracy | Evaluates structured document regions    |

---

## ⚙️ Tech Stack

| Technology                | Usage                                |
| ------------------------- | ------------------------------------ |
| Python 3.10+              | Core development                     |
| PyTorch                   | Model training                       |
| Hugging Face Transformers | LayoutLMv3 implementation            |
| Datasets                  | FUNSD dataset loading                |
| Evaluate / seqeval        | Evaluation metrics                   |
| PyMuPDF                   | PDF text and bounding-box extraction |
| Pillow                    | Image processing                     |
| Matplotlib                | Visualization                        |

---

## 🧠 Model and Dataset

**Base Model:** `microsoft/layoutlmv3-base`

**Dataset:** `nielsr/funsd-layoutlmv3`

**Task:** Token Classification

### Label Space

```text
O
B-QUESTION
I-QUESTION
B-ANSWER
I-ANSWER
B-HEADER
I-HEADER
```

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2️⃣ Install Dependencies

```bash
pip install -q seqeval pymupdf transformers datasets evaluate jiwer Pillow torch torchvision
```

### 3️⃣ Open Notebook

Open the notebook in **Jupyter** or **VS Code Notebook**:

```text
file.ipynb
```

### 4️⃣ Run Cells in Order

Execute the notebook from top to bottom to:

* Prepare the FUNSD dataset
* Fine-tune LayoutLMv3
* Evaluate the model
* Process PDF documents
* Generate structured JSON
* Create visualization outputs

---

## 🖥️ How It Works

1. **FUNSD** is loaded and document labels are mapped to BIO tags.
2. A custom PyTorch dataset aligns words, bounding boxes, and labels.
3. **LayoutLMv3** is fine-tuned for document token classification.
4. Baseline and fine-tuned models are evaluated using precision, recall, and F1.
5. **PyMuPDF** extracts words and bounding boxes from PDF documents.
6. The fine-tuned model predicts labels and confidence scores for each token.
7. Contiguous BIO predictions are merged into structured document regions.
8. Predictions are exported as **JSON** and visualization images.

---

## 🔍 Pipeline Architecture

```text
PDF / Image
     │
     ▼
PyMuPDF Word + Bounding Box Extraction
     │
     ▼
LayoutLMv3 Token Classification
     │
     ▼
BIO Labels + Confidence
     │
     ▼
Region Grouping
     │
     ▼
Structured JSON
     │
     ▼
Visualization
```

---

## 📊 Training Configuration

| Parameter                | Value              |
| ------------------------ | ------------------ |
| Epochs                   | 10                 |
| Encoder Learning Rate    | `5e-5`             |
| Classifier Learning Rate | `2.5e-4`           |
| Weight Decay             | `0.01`             |
| Batch Size               | `2`                |
| Max Sequence Length      | `512`              |
| Warmup                   | First 10% of steps |
| Model                    | LayoutLMv3 Base    |

---

## 📄 Output Schema

The parser converts token-level predictions into region-level JSON:

```json
{
  "page": 0,
  "regions": [
    {
      "type": "HEADER",
      "text": "EMPLOYEE INFORMATION FORM",
      "bounding_box": [84, 55, 640, 104],
      "confidence": 0.93
    }
  ]
}
```

---

## 📁 Project Structure

```text
layout-aware-document-parser/
├── file.ipynb
├── README.md
├── benchmark_results.csv
├── parsed_document.json
├── funsd_ground_truth.png
├── predictions_vs_ground_truth.png
├── full_pipeline_output.png
├── layoutlm_training_curves.png
└── layoutlmv3_funsd_finetuned/
```

---

## 📄 Notes

* FUNSD is used as the primary document-understanding dataset.
* LayoutLMv3 uses textual, visual, and spatial document information.
* PyMuPDF extracts PDF words and their bounding boxes for inference.
* Long documents are processed using sequence chunking.
* Results may vary depending on GPU, random seed, and library versions.
* Package versions and random seeds should be pinned for reproducible experiments.

---

## ⚠️ Limitations

* FUNSD is relatively small and focused on form documents.
* Generalization to unseen document layouts may require additional domain adaptation.
* Chunk-based inference can split contextual entities across boundaries.
* The current implementation is primarily notebook-based.

---

## 🚧 Roadmap

* Add deterministic seed configuration.
* Move training and evaluation into standalone Python scripts.
* Add ONNX/TorchScript export for deployment.
* Add document-level post-processing.
* Improve handling of long documents and cross-chunk entities.

---

## 👨‍💻 Author

**Vashishtha Verma**

* 🤖 Machine Learning & Generative AI
* 🧠 NLP & Large Language Models
* 📄 Document AI & OCR
* 💻 Software Engineering & DSA
