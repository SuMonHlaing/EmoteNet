# EmoteNet: Facial Emotion Recognition with ResNet + CBAM

A deep learning project for facial emotion recognition using ResNet architecture enhanced with Convolutional Block Attention Modules (CBAM). This model classifies facial expressions into 7 emotion categories.

## 📋 Overview

EmoteNet combines ResNet's powerful feature extraction with CBAM attention mechanisms to improve emotion recognition accuracy. The project uses the RAF-DB (Real-world Affective Faces Database) dataset for training and evaluation.

## 🎯 Features

- **ResNet-based Architecture**: Leverages pre-trained ResNet50 for robust feature extraction
- **CBAM (Convolutional Block Attention Module)**: Adds channel and spatial attention for improved focus on emotional regions
- **7 Emotion Classes**: Detects Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
- **Data Augmentation**: Random horizontal flips and normalization for better generalization
- **Dropout Regularization**: Prevents overfitting with configurable dropout rates

## 📁 Dataset Structure

The project uses the RAF-DB dataset with the following structure:

```
basic/
├── Annotation/
│   ├── auto/              # Automatic annotations
│   ├── boundingbox/       # Face bounding boxes
│   └── manual/            # Manual annotations
├── EmoLabel/
│   └── list_patition_label.txt  # Train/test split with emotion labels
├── Feature/
│   ├── baseDCNN.mat       # Pre-extracted DCNNfeatures
│   ├── DLP-CNN.mat        # DLP-CNN features
│   ├── Gabor.mat          # Gabor filter features
│   └── HOG.mat            # Histogram of Oriented Gradients
└── Image/
    ├── aligned/           # Aligned face images (224x224)
    └── original/          # Original unaligned images

data/
├── train_labels.csv       # Training set labels and paths
└── test_labels.csv        # Test set labels and paths
```

**Dataset Source**: [RAF-DB Dataset](https://drive.google.com/drive/folders/1XguZY-geR_MzqVUDVbSmi6zSWi42EBYG?usp=drive_link)

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd EmoteNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities and pre-trained models
- **tqdm**: Progress bar for training loops
- **Pillow**: Image processing
- **NumPy/Pandas**: Data handling (implicit through torch ecosystem)

See `requirements.txt` for specific versions.

## 🚀 Usage

### Training

```bash
python train.py
```

**Configuration** (edit in `train.py`):
- `BATCH_SIZE`: 16 (adjust based on GPU memory)
- `EPOCHS`: 30
- `LR`: 1e-4 (learning rate)
- `NUM_CLASSES`: 7 (emotions)
- `DATA_ROOT`: Path to aligned images
- `ANNOTATION_FILE`: Path to emotion labels

### Model Architecture

The `ResEmoteNetCBAM` model consists of:
1. **ResNet50 backbone** (pre-trained on ImageNet)
2. **CBAM attention module** applied to the final residual block (2048 channels)
3. **Spatial pooling** (adaptive average pooling)
4. **Dropout** (0.5 rate) for regularization
5. **Fully connected classifier** (2048 → 7 emotions)

### Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script with data loading and model training |
| `dataset.py` | Custom PyTorch Dataset class for RAF-DB |
| `models/resemonet_cbam.py` | ResNet50 + CBAM model definition |
| `models/cbam.py` | CBAM attention module implementation |
| `utils.py` | Utility functions (if present) |

## 📊 Emotion Classes

The model predicts 7 emotion categories:
- 0: Surprise
- 1: Fear
- 2: Disgust
- 3: Happiness
- 4: Sadness
- 5: Anger
- 6: Neutral

## 🎛️ Model Details

### CBAM (Convolutional Block Attention Module)

CBAM improves ResNet by adding two complementary attention mechanisms:

**Channel Attention**: Uses MLP to recalibrate channel-wise feature responses
- Uses both average and max pooling
- Learned through a bottleneck structure with reduction ratio = 16

**Spatial Attention**: Generates attention maps across spatial dimensions
- Uses concatenated average and max pooled features
- Applies convolution (kernel=7) and sigmoid activation

### Regularization

- **Dropout (p=0.5)**: Applied before the final classifier to prevent overfitting
- **Data Augmentation**: Random horizontal flips during training
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## 💾 Large Files & Git LFS

The `basic/` and `data/` folders contain large dataset files tracked using Git LFS:
- `.mat` files (pre-extracted features)
- `.zip` files (images)
loremjk

If cloning, ensure Git LFS is installed:
```bash
git lfs install
git clone <repository-url>
git lfs pull
```

## 🔄 Data Flow

```
Input Image (224×224) 
    ↓
ResNet50 Features (2048×7×7)
    ↓
CBAM Attention (channel + spatial)
    ↓
Adaptive Average Pooling (2048)
    ↓
Dropout (p=0.5)
    ↓
FC Layer → 7 Emotions
```

## 🛠️ Troubleshooting

**Issue**: "No images found in {root_dir}"
- Verify `DATA_ROOT` path points to `basic/Image/aligned/`
- Check that image filenames end with `_aligned.jpg`
- Ensure annotation file has correct format: `filename.jpg label`

**Issue**: CUDA out of memory
- Reduce `BATCH_SIZE` in `train.py`
- Consider using gradient checkpointing (requires model modifications)

**Issue**: Low accuracy
- Increase `EPOCHS` or adjust `LR`
- Verify dataset labels are 0-indexed (handled in `dataset.py`)
- Check image quality and alignment

## 📈 Future Improvements

- [ ] Multi-task learning with auxiliary branches (age, gender prediction)
- [ ] Ensemble methods combining multiple architectures
- [ ] Cross-domain adaptation for other emotion datasets
- [ ] Real-time inference on video streams
- [ ] Explainability analysis (attention heatmaps, grad-CAM)

## 📝 License

[Add your license here]

## 👥 Authors

Created for facial emotion recognition research and applications.

## 📖 References

- RAF-DB Dataset: [ResearchGate RAF-DB](https://www.researchgate.net/publication/318407527)
- CBAM Paper: [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- ResNet Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

**Last Updated**: January 2026
