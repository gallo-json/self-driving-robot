import sys
sys.path.insert(1, '/home/jose/Programming/aiml/tools/yolov3-archive')

from models import Darknet, load_darknet_weights
from torchsummary import summary
import torch

model = Darknet('/home/jose/Programming/aiml/Projects/self-driving-robot/src/yolo/stop-sign/conf/yolov4-custom-for-torch.cfg').cuda()
model.load_state_dict(torch.load('/home/jose/Programming/aiml/Projects/self-driving-robot/src/yolo/stop-sign/weights/yolov4-custom-for-torch_best.pt')['model'])

summary(model, input_size=(3, 416, 416))