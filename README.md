# Plant-Detection using YOLOn11
A deep learning-based computer vision project to detect and classify plant species using the YOLOv8 model. The model was trained on a custom Roboflow dataset with the aim of accurately detecting common indoor plants.

This project was created as part of a club initiative to experiment with object detection models and custom datasets.
# 🚀 Features
✅ Custom-trained YOLOv8 model
✅ Detects multiple types of indoor plants
✅ Lightweight and efficient for real-time use
✅ Ready-to-use for further finetuning, validation, or deployment
✅ Outputs include predictions, validation plots (confusion matrix, PR curve, F1 curve)


# 📂  Dataset
Source: Dataset- https://universe.roboflow.com/krishna-1l7vd/plant-detection-926d9/dataset/1

Format: YOLO11n

Version: 1

Classes:

=>Money Plant

=>Aloe Vera

=>Rattlesnake Plant

=>Begonia

👉 Annotations were carefully prepared. Some classes such as Aloe Vera required improved augmentation strategies due to slightly lower detection scores.

# 🧠 Model
Model Used: YOLO11n (YOLO11n nano variant for lightweight fast inference)

Framework: Ultralytics YOLOv8 (Python 3.11, Torch 2.6, CUDA on Tesla T4)

Epochs: 15

Image Size: 640×640

Best Weights File: best.pt (saved automatically by Ultralytics YOLO11n after training)

# 📂 Outputs
🔹 Best weights: saved at runs/detect/train/weights/best.pt

🔹 Validation results: plots saved under runs/detect/train/

   confusion_matrix.png

   PR_curve.png

   F1_curve.png

🔹 Inference predictions: output images saved under runs/detect/predict*/

# Example Outputs

![WhatsApp Image 2025-06-16 at 17 21 15_ef6fc9bd](https://github.com/user-attachments/assets/fc8d2176-cc77-43bb-be93-72302c3bef35)

![WhatsApp Image 2025-06-16 at 17 21 23_70a65585](https://github.com/user-attachments/assets/115b9db3-a48e-43ab-b101-89b1d4c2624e)

![WhatsApp Image 2025-06-16 at 17 21 32_79830e0a](https://github.com/user-attachments/assets/eca57857-46a4-426d-9d20-75e99cc1107c)

![Screenshot 2025-06-16 172122](https://github.com/user-attachments/assets/a2fa0ca2-dfc6-46e0-8006-dca1e40b8266)

![WhatsApp Image 2025-06-16 at 17 24 51_fe5cb30b](https://github.com/user-attachments/assets/6553b79c-192e-412e-abf6-42c9b1b9052d)


# 📝 Insights:

Begonia has excellent accuracy – both bounding box and confidence levels are top-tier.

Aloe Vera has slightly lower scores — may need better annotation or more diverse images.

Money Plant has good detection but lower mAP@0.95 – bounding box placement could be improved.

Rattlesnake Plant is well detected with strong precision and decent generalization.

# 💾 Download
➡ Download best.pt 
➡ Download outputs: You can zip and download runs/detect/ folder for all results

# 📦 Installation

```bash
pip install ultralytics
pip install roboflow
pip install opencv-python
pip install matplotlib


## Run Example
from ultralytics import YOLO  

model = YOLO("yolo11n.pt")  

# Train
model.train(data='plant_detection/data.yaml', epochs=15, imgsz=640)

# Validate
model.val(data='plant_detection/data.yaml', save=True)

# Predict
model.predict(source='test_images/', save=True, conf=0.25)





