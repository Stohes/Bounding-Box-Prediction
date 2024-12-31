This project focuses on fine-tuning the YOLO 11 model, chosen for its exceptional speed and suitability for deployment on NVIDIA Jetson boards, where it achieves both high accuracy and fast inference. 

The model delivers an impressive average inference runtime of 15.5 ms, making it ideal for real-time applications.

All the code utilized for this project is consolidated in the `probe_detection.ipynb` notebook, ensuring an organized and comprehensive workflow.

Validation results are available in `runs/detect/probe_val/`, while the trained model weights are stored in `runs/probe_train/weights/`. 

Future improvements may include hyperparameter tuning using AWS SageMaker to optimize performance further and integrating TensorRT to reduce inference time even more on the Jetson board.
