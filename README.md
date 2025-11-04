# MLE_PCB_defect_detection
Our early version of PCB defect detection with YOLOv8 for the course ML for EEE 

# 🔍 PCB Defect Detection using YOLOv8

This project implements an **automated Printed Circuit Board (PCB) defect detection system** using the **YOLOv8** object detection model.  
It identifies six major types of PCB defects from images with high accuracy, even on limited hardware resources.

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

### 1️⃣ Clone and install dependencies

git clone https://github.com/yourusername/PCB-Defect-Detection.git
cd PCB-Defect-Detection

pip install ultralytics gradio opencv-python-headless matplotlib

## 🧩 Model Training
Model Configuration
Parameter	Value
Base model	yolov8n.pt (Nano)
Image size	416 × 416
Batch size	2
Epochs	40
Mosaic	0.4
Mixup	0.0
Save period	10
AMP	Enabled
Device	NVIDIA RTX 3050 Ti (4 GB)

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
