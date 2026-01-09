# EmoteNet - Facial Emotion Recognition

ResNet50 + CBAM attention model for classifying facial expressions into 7 emotion categories using RAF-DB dataset.

## Quick Start

```bash
git clone https://github.com/SuMonHlaing/EmoteNet
cd EmoteNet
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Dataset

- **Source**: [RAF-DB](https://drive.google.com/drive/folders/1XguZY-geR_MzqVUDVbSmi6zSWi42EBYG?usp=drive_link)
- **Location**: `basic/Image/aligned/` (224×224 aligned faces)
- **Labels**: `basic/EmoLabel/list_patition_label.txt`
- **Emotions**: 0=Surprise, 1=Fear, 2=Disgust, 3=Happiness, 4=Sadness, 5=Anger, 6=Neutral

## Model

- **Backbone**: ResNet50 (ImageNet pre-trained)
- **Attention**: CBAM (channel + spatial)
- **Regularization**: Dropout (0.5)
- **Output**: 7 emotion classes

## Training Config

Edit `train.py`:
```python
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DATA_ROOT = "basic/Image/aligned"
ANNOTATION_FILE = "basic/EmoLabel/list_patition_label.txt"
```

## Project Structure

```
├── train.py              # Training script
├── dataset.py            # PyTorch Dataset
├── models/
│   ├── resemonet_cbam.py # ResNet + CBAM
│   └── cbam.py           # CBAM module
├── basic/                # Dataset (Git LFS)
├── data/                 # Train/test data
└── requirements.txt
```

## Clone with Git LFS

```bash
git lfs install
git clone <repository-url>
git lfs pull
```

## Dependencies

- torch
- torchvision
- tqdm

---

Last Updated: January 2026
