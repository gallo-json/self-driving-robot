import sys
sys.path.insert(1, '/home/jose/Programming/aiml/tools/yolov3-archive')

from models import Darknet
from utils.datasets import *
from utils.utils import torch_utils, load_classes

infer_image_size = 224
cfg = '/home/jose/Programming/aiml/Projects/self-driving-robot/src/yolo/stop-sign/conf/yolov4-custom-for-torch.cfg'
weights = '/home/jose/Programming/aiml/Projects/self-driving-robot/src/yolo/stop-sign/weights/yolov4-custom-for-torch_best.pt'
names = '/home/jose/Programming/aiml/Projects/self-driving-robot/src/yolo/stop-sign/conf/obj.names'
confidence_thres = 0.3
iou_thres = 0.6

# Select CUDA device
device = torch_utils.select_device()

# Load the YOLO model
model = Darknet(cfg)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()

# Load the class names
names = load_classes(names)

# Choose random colors for bounding box
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

### Start INFFERENCE
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

t0 = time.time()

# Initialize dummy image
img = torch.zeros((1, 3, infer_image_size, infer_image_size), device=device)
_ = model(img.float())

while True:
    ok, img = cap.read()

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0

    if img.ndimension() == 3: img = img.unsqueeze(0)

    # Actual inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=False)[0]
    t2 = torch_utils.time_synchronized()

    pred = non_max_suppression(pred, confidence_thres, iou_thres, multi_label=False, agnostic=False)