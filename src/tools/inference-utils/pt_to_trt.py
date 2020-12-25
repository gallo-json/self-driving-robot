# This file references code from the repo https://github.com/ultralytics/yolov3/tree/archive but it is not part of it
# Run this file inside that cloned repo to get no errors
# torch2trt was made primarily for the Jetson Nano, run it on that.

from models import *
from torch2trt import torch2trt
import torch

model = Darknet('/home/jetbot/self-driving-robot/src/yolo/stop-sign/conf/yolov4-custom-for-torch.cfg').eval().cuda()
model.load_state_dict(torch.load('/home/jetbot/self-driving-robot/src/yolo/stop-sign/weights/yolov4-custom-for-torch_best.pt')['model'])

x = torch.ones((1, 3, 416, 416)).cuda()

model_trt = torch2trt(model, [x])

torch.save(model_trt.state_dict(), 'yolov4.trt')
