# Self Driving Robot

## Powered by
![](resources/tech-stack.png)

## Inference
### Directly on the Jetson Nano

Inference with only the darknet `.weights` file loaded by OpenCV performs very poorly (~0.8 FPS). To better the performance convert the Darknet YOLOv4 model to a PyTorch model or to a TensorRT engine.

### On the computer SSHing into the Jetson Nano

If worst comes to worst and the Nano cannot run the model smoothly, I can use my NVIDA GPU laptop that controls the Nano. The Nano sends the live camera feed to the laptop, the laptop does all the inference then sends back the labels where then the Nano parses that and moves the motors accordingly.

## Road following

NVIDIA AI IOT already has source code for this. All that is left is to combine both models.

## Useful repositories

JetBot: https://github.com/NVIDIA-AI-IOT/jetbot
TensorRT demos: https://github.com/jkjung-avt/tensorrt_demos
YOLO in PyTorch: https://github.com/ultralytics/yolov3/tree/archive
