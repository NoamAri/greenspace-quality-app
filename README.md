# 🌳 VLM Sense - Greenspace Quality Classifier

## Deployment-Ready Application

This folder contains **ONLY** the essential files needed to run the Greenspace Quality Classifier app.

---

## 📁 What's Included

### Core Application
- **`app.py`** - Main Streamlit web interface
- **`config.py`** - Configuration settings

### AI Components
- **`vegetation_detector.py`** - Vegetation detection (HSV color-based)
- **`vegetation_features.py`** - Feature extraction from vegetation regions
- **`scene_features.py`** - CLIP-based scene understanding
- **`dataset.py`** - Image preprocessing utilities

### Trained Models
- **`models/`** folder containing:
  - `best_classifier.pkl` - Trained Random Forest classifier
  - `scaler.pkl` - Feature normalization
  - `confusion_matrix.png` - Model evaluation

### Deployment Files
- **`requirements.txt`** - Python dependencies
- **`packages.txt`** - System dependencies (for Streamlit Cloud)
- **`.gitignore`** - Git ignore rules

### Documentation
- **`System_Architecture_Documentation.html`** - Full system documentation

---

## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `app.py` as the main file
5. Deploy!

---

## 📊 What the App Does

Analyzes park/greenspace images and classifies them as:
- **Healthy** (vibrant green vegetation)
- **Dried** (yellow/brown vegetation)
- **Contaminated** (polluted/degraded areas)

### Features:
- ✅ Lightweight color-based vegetation detection
- ✅ CLIP embeddings for scene understanding
- ✅ Random Forest classifier
- ✅ Optimized for cloud deployment (1GB RAM limit)

---

## 🧠 System Requirements

**Minimum:**
- Python 3.8+
- 1GB RAM (cloud)
- 4GB RAM (local for better performance)

**Cloud Optimizations Applied:**
- Auto-resize images to max 1024px
- Disabled heavy GroundingDINO model
- Uses lightweight HSV color detection
- Single-threaded CPU operation

---

## 📖 For More Details

Open `System_Architecture_Documentation.html` in your browser for:
- Complete file structure explanation
- Data flow diagrams
- Model descriptions
- Feature engineering details

---

## ⚠️ What's NOT Included

The following files were excluded (used only for development/training):
- Training scripts (`train_classifier.py`, `extract_features.py`)
- Visualization tools (`visualize_*.py`)
- Test scripts (`test_*.py`)
- Training data (`Data/` folder)
- Extracted features (`features/` folder)
- GroundingDINO weights (not used in lightweight mode)

**You can safely delete everything else outside this folder!**

---

## 🔧 Quick Troubleshooting

### App won't start
- Check Python version: `python --version` (needs 3.8+)
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Out of memory on cloud
- Images are auto-resized to 1024px
- If still crashing, reduce `max_size` in `app.py` line 95

### "Models not found" error
- Ensure `models/` folder exists with `.pkl` files
- Check file paths in `config.py`

---

**Version:** January 2026  
**Status:** Production-Ready ✅
