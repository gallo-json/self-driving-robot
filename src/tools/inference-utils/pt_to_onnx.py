import sys
sys.path.insert(1, '/home/jetbot/yolov3-archive')

import torch
import torch.onnx
import torchvision
import torchvision.models as models
from models import Darknet, load_darknet_weights

model = Darknet('/home/jetbot/self-driving-robot/src/yolo/stop-sign/conf/yolov4-custom-for-torch.cfg')
model.load_state_dict(torch.load('/home/jetbot/self-driving-robot/src/yolo/stop-sign/weights/yolov4-custom-for-torch_best.pt')['model'])

onnx_model_path = "yolo.onnx"
 
# set the model to inference mode
model.eval()
 
# Create some sample input in the shape this model expects 
# This is needed because the convertion forward pass the network once 
dummy_input = torch.randn(1, 3, 416, 416)
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)
