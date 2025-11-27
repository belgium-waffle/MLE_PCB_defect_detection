
# 🔍 PCB Defect Detection using YOLOv8  
### BITS Pilani — Machine Learning for EEE (MLE) Course Project  
**Team Members:**  
- Jaikumar Wath (2023A3PS0197G)  
- Aditya Mane (2023A3PS0505G)  
- Vaibhav Divakar (2023A3PS0527G)

---

## 📌 Project Overview

This repository contains our full pipeline for **automated PCB defect detection** using **YOLOv8**.  
The goal is to reliably detect six types of microscopic PCB manufacturing defects:

- Missing hole  
- Mouse bite  
- Open circuit  
- Short  
- Spur  
- Spurious copper  

The project includes:

- 📁 Complete dataset preprocessing pipeline  
- 🔄 XML → YOLO format annotation conversion  
- 🧠 Training using YOLOv8m (768×768)  
- 🔃 3-Fold Cross Validation  
- 📊 Training curves + detailed metrics  
- 🌐 Deployment as a **Gradio Web Application**  
- 📦 Released final trained model  

---

## 📁 **Repository Structure**

- **Website/**
  - Gradio Web App (final deployed product)

- **example_images/**
  - Sample accurate model output images for you to see 

- **final_results/**
  - Final validation metrics, loss curves, training plots, and prediction screenshots

- **midsem/**
  - Mid-semester submission (baseline code, report, and intermediate results)

- **git_vs_us/**
  - Comparison between public GitHub YOLOv8 training scripts and our optimized pipeline
  - Includes YOLOv8m vs YOLOv8s comparison experiments

- **finalyolov8m.ipynb**
  - Main training notebook (YOLOv8m at 768×768 with 3-Fold Cross-Validation)

- **README.md**
  - Full project documentation (this file)

- **Release/Final_Trained_Model/**
  - Exported final model weights (`best.pt`, `best.torchscript`)


---

## 📦 Dataset Information

We use the **PCB Defects** dataset released by the  
*Open Lab on Human–Robot Interaction (Peking University)*.

Dataset link:  
🔗 **https://www.kaggle.com/datasets/akhatova/pcb-defects/data**

Dataset structure (per class):



PCB_DATASET/
├── images/
│ ├── Missing_hole/
│ ├── Mouse_bite/
│ ├── Open_circuit/
│ ├── Short/
│ ├── Spur/
│ └── Spurious_copper/
└── Annotations/ (Pascal VOC XML)


We parse all VOC XML annotations and convert them into normalized YOLOv8 `.txt` format.

---

## 🔧 **Key Features of This Project**

### ✔ 1. **Custom XML → YOLO Converter**
We implemented a scalable XML parsing pipeline using `xml.etree.ElementTree`, converting:



xmin, ymin, xmax, ymax → (class, x_center, y_center, width, height)


Fully automatic, supports thousands of images.

---

### ✔ 2. **Correct Letterbox Preprocessing (No Distortion)**  
Unlike naive resizing, our preprocessing:

- preserves aspect ratio  
- pads remaining space with 114  
- applies the same affine transform to bounding boxes  

This matches Ultralytics behavior and improves mAP noticeably.

---

### ✔ 3. **Train/Val/Test Split + K-Fold Cross Validation**

We generate:



train/
val/
test/


and additionally perform **3-fold CV** to evaluate model robustness.

Each fold gets its own:

- images/
- labels/
- dataset.yaml  

---

### ✔ 4. **YOLOv8m Training with Advanced Augmentations**

We train **YOLOv8m (25M params)** at **768×768** resolution using Kaggle GPU (T4/P100).  
Augmentations include:

- Mosaic = 0.8  
- RandAugment  
- Copy-Paste  
- HSV jitter  
- Perspective = 0.0005  
- Erasing = 0.35  

These improve tiny-defect detection significantly.

---

### ✔ 5. **Final Performance (Fold-2 Best Model)**

| Metric        | Score |
|---------------|-------|
| Precision     | **0.966** |
| Recall        | **0.962** |
| mAP@50        | **0.972** |
| mAP@50–95     | **0.529** |

Per-class AP values show strong performance across all 6 defect types.

---

### ✔ 6. **Gradio Web Application (Productization)**

We convert our model into a **real, usable tool**:

🔗 *`Website/app.py`* contains a Gradio interface:

- upload a PCB image  
- YOLOv8 runs inference  
- annotated image is returned instantly  

This demonstrates practical deployability for factory QC lines.

---

## 🚀 **How to Run the Project**

### 1. Install dependencies

```pip install ultralytics gradio opencv-python-headless matplotlib pandas```

2. Download dataset

Upload the dataset in the same structure as described above.

3. Run the preprocessing + training notebook
jupyter notebook finalyolov8m.ipynb

4. Run the final trained model
from ultralytics import YOLO
model = YOLO("final_model/best.pt")
model.predict("example_images/short_01.jpg")

5. Launch the Gradio Web App
cd Website
python app.py


The interface will be available at:

http://localhost:7860

---

## 📥 Download Final Model

The final trained model is provided as part of the GitHub release bundle:

**Included:**
- ✔ `best.pt` (PyTorch YOLOv8 format)

You can download it from:

👉 **Releases → Final Trained Model**

---

## 🧪 Training Curve Examples

Training logs, epoch-wise metrics, and loss curves are available in:

final_results/


This folder includes:
- Box loss vs. epochs  
- Class loss vs. epochs  
- Distribution focal loss  
- Validation mAP curves  
- Sample predictions  

---

## 📚 Technologies Used

This project was built using:

- **Python 3.10**
- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **OpenCV**
- **NumPy / Pandas**
- **Matplotlib**
- **Gradio (Web Deployment)**
- **Kaggle GPU Compute (T4 / P100)**

---

## 👨‍💻 Contributions

All team members contributed equally across the following components:

- Dataset preprocessing & organization  
- XML parsing and YOLO label generation  
- Model training and hyperparameter tuning  
- K-Fold cross-validation setup  
- Inference visualization & evaluation  
- Gradio web interface development  
- Report writing and documentation  

---

## 📄 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute this project with proper attribution.

---

## ⭐ Support

If you found this project useful, please consider giving the repository a **⭐ star** on GitHub.  
It helps others discover the project and supports our work.

---
