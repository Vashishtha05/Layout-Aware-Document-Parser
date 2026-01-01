# Layout Aware Document Parser

A complete notebook-based pipeline for layout-aware document understanding using LayoutLMv3, trained on FUNSD, and applied to PDF parsing with PyMuPDF.

## Overview

This project fine-tunes `microsoft/layoutlmv3-base` for token classification on form documents. The notebook demonstrates the full workflow:

1. Load and inspect the FUNSD dataset.
2. Build a custom PyTorch dataset that aligns words, bounding boxes, and labels.
3. Fine-tune LayoutLMv3 for entity tagging (`QUESTION`, `ANSWER`, `HEADER`, `O`).
4. Evaluate performance before and after fine-tuning.
5. Run an end-to-end PDF pipeline:
   - Extract words and bounding boxes with PyMuPDF.
   - Classify each token with the fine-tuned model.
   - Convert predictions into structured JSON regions.
   - Visualize results.

## Key Features

- End-to-end training + inference in a single notebook.
- Layout-aware modeling with text, image, and 2D spatial features.
- Baseline vs fine-tuned benchmark reporting.
- Region-level structured output (`parsed_document.json`).
- Visual diagnostics for model behavior.

## Tech Stack

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- Datasets + Evaluate
- seqeval
- PyMuPDF (`fitz`)
- Pillow
- Matplotlib

## Model and Dataset

- Base model: `microsoft/layoutlmv3-base`
- Fine-tuning dataset: `nielsr/funsd-layoutlmv3`
- Task type: token classification
- Label space:
  - `O`
  - `B-QUESTION`, `I-QUESTION`
  - `B-ANSWER`, `I-ANSWER`
  - `B-HEADER`, `I-HEADER`

## Pipeline Architecture

```text
PDF/Image -> OCR Words + Boxes -> LayoutLMv3 Token Classification -> BIO Labels -> Region Grouping -> Structured JSON
```

Detailed flow in notebook:

1. Data prep: FUNSD loading and label mapping.
2. Training: fine-tuning LayoutLMv3 with differential learning rates.
3. Evaluation: seqeval metrics (`precision`, `recall`, `F1`).
4. Inference:
   - Chunk long documents.
   - Predict token labels + confidence.
   - Merge contiguous BIO spans into entity regions.
5. Output generation:
   - Benchmark CSV
   - Parsed JSON
   - Visualization PNG files

## Repository Structure

```text
Layout Aware Document Parser/
  file.ipynb
  README.md
```

Generated after running the notebook:

```text
benchmark_results.csv
parsed_document.json
funsd_ground_truth.png
predictions_vs_ground_truth.png
full_pipeline_output.png
layoutlm_training_curves.png
layoutlmv3_funsd_finetuned/
```

## Quick Start

### 1) Clone repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2) Open notebook

Open `file.ipynb` in Jupyter or VS Code Notebook.

### 3) Install dependencies

The notebook includes install cells. If you prefer local install:

```bash
pip install -q seqeval pymupdf transformers datasets evaluate jiwer Pillow torch torchvision
```

### 4) Run cells in order

Execute all cells top to bottom to:

- fine-tune the model,
- evaluate benchmark metrics,
- generate parser outputs and visualizations.

## Training Configuration (Current Notebook)

- Epochs: 10
- Learning rate (encoder): `5e-5`
- Learning rate (classifier head): `2.5e-4`
- Weight decay: `0.01`
- Batch size: `2`
- Max sequence length: `512`
- Warmup schedule: linear warmup for first 10% of total steps

## Inference and Output Schema

The parser converts token-level predictions into region-level JSON objects:

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

## Reproducibility Notes

- Results depend on GPU type, seed behavior, and library versions.
- For stable comparisons, pin package versions and set random seeds.
- The notebook currently uses runtime-local paths for some outputs.

## Limitations

- FUNSD is a relatively small dataset.
- Generalization to unseen document templates may require domain adaptation.
- Chunk-based inference can split contextual spans across chunk boundaries.

## Roadmap

- Add deterministic seed setup across Python, NumPy, and PyTorch.
- Add script-based training and evaluation outside notebook.
- Add ONNX or TorchScript export for production deployment.
- Add document-level post-processing heuristics.

## License

Add your preferred license (for example, MIT) before public release.

## Acknowledgements

- Hugging Face Transformers and Datasets
- Microsoft LayoutLMv3
- FUNSD dataset authors
- PyMuPDF maintainers
