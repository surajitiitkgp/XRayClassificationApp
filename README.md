# Chest X-ray Classification GUI

A **PyQt5-based graphical interface** for classifying chest X-ray images (DICOM format) using **pretrained models from TorchXRayVision**.  

This tool allows users to:  
- Load X-ray DICOM images from a folder  
- Select pretrained DenseNet121 weights (CheXpert, NIH, RSNA, MIMIC, etc.)  
- Choose a pathology to check  
- Run classification with probabilities  
- Visualize **SmoothGrad-CAM++ heatmaps**  
- Save structured results in CSV  

---

## ✨ Features
- 📂 DICOM folder selection  
- ⚙️ Pretrained weight options (`densenet121-res224-*`)  
- 🩻 Pathology dropdown (auto-populated from model)  
- 🔥 SmoothGrad-CAM++ heatmap visualization  
- 📊 Probability scores + Top-3 associated diseases  
- 💾 Export results to CSV  
- ✅ Accuracy calculation per folder  
- 🖼️ Scrollable viewer for overlayed heatmaps  

---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/surajitiitkgp/XRayClassificationApp.git
cd XRayClassificationApp

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

pip install torch torchvision pydicom torchxrayvision torchcam PyQt5 matplotlib pandas

python XRayClassificationApp.py


