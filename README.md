# MediVision Multimodal Health Suite — preservation update

This project is a conservative update of the supplied `app.py` and `medical_report_analyzer.py`. The original visual theme, three-module navigation, report explanations, configured lab ranges, CSV download, and chest DenseNet121 architecture/pipeline are retained. The kidney module now uses the intended Keras four-class classifier and refuses to predict until its actual training preprocessing and class-index order are configured.

## Required model files

Place these files in `models/`:

1. `models/best_model.keras` — new Keras kidney model.
2. `models/densenet121_3class_best.pth` — existing chest model checkpoint.
3. `models/kidney_model_config.json` — verified metadata for the Keras model, populated from the notebook.

The uploaded `kidney_c.ipynb` verifies these settings:
- Output index 0 = `Cyst`, 1 = `Normal`, 2 = `Stone`, 3 = `Tumor`. The notebook sets `CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]` and labels each class using `enumerate(CLASSES)`.
- Input size is `380 x 380`, RGB.
- Images are resized and cast to `float32` while retaining the `0–255` pixel range.
- The model is EfficientNetB4 with a four-unit softmax output. EfficientNetB4 contains internal input rescaling, so the app must not add a second external normalization step.
- The notebook reports TensorFlow `2.21.0`.

The verified `models/kidney_model_config.json` is included. Put your trained file at `models/best_model.keras` (the notebook also saves `kidney_efficientnetb4_final.keras`; either file should be renamed to `best_model.keras` if it is the intended trained model). The model's output shape is checked to be `(batch, 4)`. A class score is not a clinical probability or diagnosis.

## Windows local setup

Use Python 3.11 for the current combined TensorFlow/PyTorch dependency set. From PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Install Tesseract OCR separately on Windows and ensure `tesseract.exe` is on PATH. The analyzer also checks common Windows install paths. `pytesseract` alone does not install the executable.

## Streamlit Community Cloud deployment

1. Push this source to the existing GitHub repository in a separate branch first; keep the currently deployed version unchanged until tested.
2. In Streamlit Community Cloud, select the repository, branch, and `app.py` as the entry point (or update the existing app's branch only after testing).
3. Commit `requirements.txt` and root-level `packages.txt` (`tesseract-ocr`).
4. Upload model files to `models/` using Git LFS or secure, reliable model hosting if they are too large for ordinary Git. Ensure deployment can access them without embedding tokens in source code.
5. Keep the included `models/kidney_model_config.json`; it reflects the uploaded notebook's class order and input preprocessing.
6. Deploy/reboot, then test all three modules using non-sensitive sample data.

I cannot inspect or change your Streamlit Cloud app settings from this source package; manually verify repository, branch, entry point, secrets, and model-file availability.

## Feature preservation checklist

- [x] Original Streamlit layout/theme and three module choices retained.
- [x] Kidney upload and results workspace retained; old binary segmentation logic replaced by four-class Keras classifier integration.
- [x] Kidney output shape is checked; missing/unverified config fails closed.
- [x] DenseNet121 checkpoint loading approach, class mapping fallback, ImageNet normalization, 224×224 preprocessing, confidence display, and class chart retained; output dimensions validated.
- [x] Existing report aliases, reference-range dictionary, explanations in English/Hindi/Marathi, rule-based findings, table, and CSV download retained.
- [x] PDF text extraction first; scanned/short-text PDFs and images can use OCR with explicit Tesseract dependency errors.
- [x] Uploaded report temporary file cleanup remains in `finally`.
- [ ] Real kidney inference — not verified because the model and training metadata were not supplied.
- [ ] Real chest inference — not verified because the checkpoint was not supplied.
- [ ] Streamlit Cloud deployment — not tested or changed.
- [ ] Exact class-index order and kidney preprocessing — not available in supplied files; must be verified from training artifacts.

## Tests

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app.py medical_report_analyzer.py
```

Tests cover parser extraction basics, unsupported/empty input, and output-shape validation. They do not prove clinical performance or real model inference.

## Troubleshooting

- **Kidney model missing:** copy the trained `.keras` file from Colab into `models/best_model.keras`; the model binary is not included in this source ZIP.
- **Kidney model output isn't four classes:** the Keras model output shape must be `(None, 4)`. Do not reinterpret a binary output.
- **Chest checkpoint load error:** ensure the original checkpoint is the trained DenseNet121 state dictionary expected by the supplied loader. Its original checkpoint and training class mapping were not included for verification.
- **Tesseract not found:** install Tesseract itself and add it to PATH; for Cloud, keep `tesseract-ocr` in `packages.txt`.
- **PDF unreadable:** use a non-password-protected PDF with selectable text or a clear scan.
- **Large models:** use Git LFS or model hosting; do not place access tokens in code.

## Medical safety

MediVision is an educational prototype, not a validated diagnostic system. OCR can misread values; configured reference ranges are simplified and may not match a specific laboratory, age, sex, pregnancy status, units, or clinical context. Review original reports and images with a qualified healthcare professional.
