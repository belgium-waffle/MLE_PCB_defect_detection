
# 🔍 PCB Defect Detection using YOLOv8

This project implements an **automated Printed Circuit Board (PCB) defect detection system** using the **YOLOv8** object detection model.  
It identifies six major types of PCB defects from images with high accuracy, even on limited hardware resources.

## 🧠 Abstract

Printed Circuit Boards (PCBs) form the backbone of all modern electronics.  
However, even a minor defect — like a missing hole, short, or open circuit — can render an entire device unusable.This project leverages **YOLOv8 (You Only Look Once)** for **automated, real-time defect detection** across multiple PCB defect types. The system achieves high accuracy with minimal compute cost, running efficiently on consumer GPUs.

---

## 🎯 Project Objectives

1. Study common PCB defect patterns and YOLOv8 methodologies  
2. Collect and annotate the dataset (from [Kaggle PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects))  
3. Resize, normalize, and augment images using **OpenCV** for generalization  
4. Train YOLOv8 on the curated dataset  
5. Tune hyperparameters (learning rate, batch size, epochs)  
6. Evaluate with **mAP**, **Precision**, **Recall**, and **F1-score**

---

## 🧩 Model Configuration

| Parameter | Value |
|------------|--------|
| Base Model | `yolov8n.pt` *(Nano)* |
| Image Size | 416 × 416 |
| Batch Size | 2 |
| Epochs | 40 |
| Mosaic | 0.4 |
| Mixup | 0.0 |
| Save Period | 10 |
| AMP | Enabled |
| GPU | NVIDIA RTX 3050 Ti (4 GB) |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone and Install Dependencies
```bash
git clone https://github.com/yourusername/PCB-Defect-Detection.git
cd PCB-Defect-Detection

pip install ultralytics gradio opencv-python-headless matplotlib
```
---

## 🧠 Defect Classes

- Missing Hole  
- Mouse Bite  
- Open Circuit  
- Short  
- Spur  
- Spurious Copper

---


## ⚙️ Setup Instructions

Training Notes
The model was trained on six PCB defect classes.

Average VRAM usage: ~1.2 GB

Disk usage: <1 GB

Validation Accuracy:

mAP@50: ~0.45–0.55

mAP@50–95: ~0.20–0.30

## 🧪 Inference (Prediction)
Run detection on a single image:

python
Copy code
from ultralytics import YOLO
model = YOLO('pcb_light_final/train/weights/best.pt')
results = model.predict(source='test_image.jpg', imgsz=416, conf=0.25)
results[0].show()
Predicted outputs are saved in:

bash
Copy code
runs/detect/predict/
## 🌐 Gradio Web App
Launch the app
bash
Copy code
python pcb_gradio.py
App Features
Upload PCB image → get defect detections in real time

Displays bounding boxes, class labels, and confidence scores

Works locally and supports public Gradio links for sharing

python
Copy code
demo.launch(server_name="0.0.0.0", server_port=None)
Sample interface:
<img width="1920" height="1080" alt="Screenshot from 2025-11-04 08-33-45" src="https://github.com/user-attachments/assets/865a3cc8-2bd8-431a-9650-7d5f87b134ca" />


💡 Key Learnings
- YOLOv8-nano performs efficiently on low-VRAM GPUs (RTX 3050 Ti 4 GB).

- Using resized images (416×416) ensures consistent results with trained input scale.

- Gradio simplifies deployment for real-time visualization and testing.

- Careful tuning of batch size, epochs, and augmentation yields strong defect detection even with limited data.

  
