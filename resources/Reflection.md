# Reflection

## Problems I faced

### Road following was not consistent

This problem really stumped me because the NVIDIA team wrote the code for the road following, not me. Their road following script has parameters one can change such as the steering and bias values. However, after many trials I could not get these values right for my particular setup. The robot kep going off course regularly. I think it was because the outline of the road wasn't clear enough compared to the cardboard background. 

### Detecting the color from the traffic lights

The YOLO model detects what the traffic lights look like in general, but it cannot tell what color it is. To fix this problem I masked the image made by the bbox to determine where in the bbox the light was coming from. If the light was coming from the very top, it was red. If the light was coming from the middle, it was yellow. Finally, if the light was coming from the bottom, it was green. I used this method of dividing the bbox image into three sections and masking the image instead of trying to match HSV values to the three different colors because depending on the lighting of the environment that would throw everything off. 

## What I enjoyed about the project

I enjoyed the fact that I was able to make such a complicated technology such as self-driving cars into a smaller, simpler scale. I'm glad I did all the computations directly on the Jeston Nano because it is quite powerful and I used it to its full potential. It was quite rewarding at the end watching the robot detect and move around. 

## Improvements

The inference speed can definitely be improved upon. Both the road-following and the stop-sign & traffic-light models running at the same time give around 3 FPS. This speed is not ideal but I really want the inference to be done directly on the Nano that way it is "edge-computing" in a sense (all the calculations are being done on the hardware and not on the cloud). The easiest way to improve the speed would be to retrain the model for YOLOv4-tiny or YOLOv5 models that are optimized for small computers like the Jetson Nano. 

## What I learned

- Self-driving car algorithms
- Inference optimization (ONNX, TensorRT, etc)
- Converting to different types of ML frameworks
- YOLOv3, YOLOv4, and training YOLO models with DarkNet and PyTorch
- Some PyTorch
- Some OpenCV deep learning inference
- Managing hardware on a NVIDIA Jetson Nano
- ResNet architecture and NVIDIA AI JetBot line following algorithm
- Google Colab / more Jupyter Notebook & managing resources on Google Drive programmatically