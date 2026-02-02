# GastroVision Project

## Overview

GastroVision is a deep learning-powered web app for gastrointestinal disease detection using endoscopy images. It supports Random Forest model inference and a modern web interface.

## Installation & Setup (on a new device)

### 1. Clone the Repository

```
git clone https://github.com/vedant-zeus/Gastro.git
cd Gastro
```

### 2. Install Python & Dependencies

- Install Python 3.8 or newer.
- Install pip packages:

```
pip install -r requirements.txt
```

### 3. Install Git LFS (for large files)

- Download and install Git LFS: https://git-lfs.github.com
- Initialize Git LFS in your repo:

```
git lfs install
```

- Pull large files:

```
git lfs pull
```

### 4. Prepare Dataset

- Place your dataset in the `dataset/` folder, organized as:

```
dataset/
  train/
  val/
  test/
```

- Each class should be a subfolder with images inside.
- If starting from raw GastroVision, use `split_gastrovision_dataset.py` to split into train/val/test.

### 5. Convert Images to RGB (if needed)

```
python convert_images_to_rgb.py
```

### 6. Train the Model (Random Forest)

```
python app/train_rf_xgb.py
```

### 7. Evaluate the Model

```
python app/evaluate_rf_xgb.py
```

### 8. Run the Web App

```
python app/app.py
```

- Open your browser at http://127.0.0.1:5000/
- Upload an endoscopy image to get predictions.

## Notes

- For deep learning (EfficientNet), use `app/train_model.py` (ensure all images are RGB).
- For XGBoost, uncomment relevant code in scripts.
- All large files (models, images) are tracked with Git LFS.

## Troubleshooting

- If you see errors about missing large files, run `git lfs pull`.
- If you see shape errors, ensure all images are RGB and correctly sized.
- For any issues, check the scripts for comments and usage instructions.

## License

Attribution 4.0 International (CC BY 4.0)
