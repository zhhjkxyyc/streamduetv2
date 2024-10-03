import subprocess
import sys

# Install the necessary package
subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/THU-MIG/yolov10.git"])

# Import the required module
from ultralytics import YOLOv10

# Load the pretrained YOLOv10 model
model = YOLOv10.from_pretrained('jameslahm/yolov10x')

# Save the model locally
model.save('yolov10x_pretrained.pth')

print("Model downloaded and saved successfully.")
