import gradio as gr
from ultralytics import YOLO
import cv2
import os

# === Load trained YOLOv8 model ===
model_path = "/home/arm/Downloads/PCB/pcb_light/train3/weights/best.pt"  # update if needed
assert os.path.exists(model_path), f"❌ Model not found at {model_path}"
model = YOLO(model_path)

# === Detection function ===
def detect_pcb_defects(image):
    # Run YOLOv8 inference
    results = model.predict(source=image, save=False, conf=0.15, imgsz=416)
    # Extract annotated image
    annotated_img = results[0].plot()  # returns numpy array (BGR)
    
    # Convert BGR → RGB for display
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_img

# === Gradio UI ===
title = "🔍 PCB Defect Detection using YOLOv8"
description = (
    "Upload a PCB image to detect defects such as missing holes, open circuits, "
    "mouse bites, shorts, and spurious copper using a trained YOLOv8 model."
)

demo = gr.Interface(
    fn=detect_pcb_defects,
    inputs=gr.Image(type="filepath", label="Upload PCB Image"),
    outputs=gr.Image(label="Detection Result"),
    title=title,
    description=description,
    examples=[
        ["/home/arm/Downloads/PCB/example_images/pcb1.png"],
        ["/home/arm/Downloads/PCB/example_images/pcb2.png"]
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=None)

