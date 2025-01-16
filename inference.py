from ultralytics import YOLO
import os
import sys

model = YOLO("runs/detect/probe_train/weights/best.pt")

# Define path to directory containing images and videos for inference
source = sys.argv[1]

# Run inference on the source
results = model(
    source,
    # stream=True # generator of Results objects
)

os.makedirs("inference_results", exist_ok=True)

for index, result in enumerate(results):
    result.save(filename=f"inference_results/result_{index}.jpg")  # save to disk