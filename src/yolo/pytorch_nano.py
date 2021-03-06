import sys
import argparse
#repo_path = '/home/jose/Programming/aiml/tools/yolov3-archive'
repo_path = '/home/jetbot/yolov3-archive'
sys.path.insert(1, repo_path)

from models import Darknet
from utils.datasets import *
from utils.utils import *

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image

### ROAD FOLLOWING
model_road_following = torchvision.models.resnet18(pretrained=False)
model_road_following.fc = torch.nn.Linear(512, 2)
model_road_following.load_state_dict(torch.load('../road_following/best_steering_model_xy.pth'))

device = torch.device('cuda')
model_road_following = model_road_following.to(device)
model_road_following = model_road_following.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

angle = 0.0
angle_last = 0.0

def find_optimal_path(image):
    global angle, angle_last
    xy = model_road_following(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0
    '''
    angle = np.arctan2(x, y)
    pid = angle * steering_gain + (angle - angle_last) * steering_kd
    angle_last = angle

    steering_value = pid + steering_bias

    robot.left_motor.value = max(min(speed_gain + steering_value, 1.0), 0.0)
    robot.right_motor.value = max(min(speed_gain - steering_value, 1.0), 0.0)
    '''
    return (x * 224)/2, (xy[1] * 224) / 2

### YOLO
stop_sign_area_thres = 0

def detect(save_img=False):
    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    source, weights, half = opt.source, opt.weights, opt.half 
    # Initialize
    device = torch_utils.select_device(opt.device)

    # Initialize model
    model_yolo = Darknet(opt.cfg, imgsz)

    # Load weights
    model_yolo.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model_yolo.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half: model_yolo.half()

    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model_yolo(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3: img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model_yolo(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half: pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    area = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))
                    s += 'Area %s px ' % (area)

                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3f FPS)' % (s, 1 / (t2 - t1)))
            
            x, y = find_optimal_path(im0)
            print(int(x), int(y))
            cv2.circle(im0, (int(x), int(y)), 8, (0, 255, 0), 3)
            cv2.imshow(p, im0)

            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

    print('Done. (%.3f FPS)' % (1 / (time.time() - t0)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
