from models import *
from torch2trt import torch2trt
import torch

model = Darknet('/home/jetbot/self-driving-robot/src/yolo/stop-sign/conf/yolov4-custom-for-torch.cfg').eval().cuda()
model.load_state_dict(torch.load('/home/jetbot/self-driving-robot/src/yolo/stop-sign/yolov4-custom-for-torch_last.pt')['model'])

x = torch.ones((1, 3, 416, 416)).cuda()

model_trt = torch2trt(model, [x])

torch.save(model_trt.state_dict(), 'yolov4.pth')
