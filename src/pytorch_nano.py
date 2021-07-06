import sys
import argparse
repo_path = '/home/jose/Programming/aiml/tools/yolov3-archive'
#repo_path = '/home/jetbot/yolov3-archive'
sys.path.insert(1, repo_path)

from models import Darknet
from utils.datasets import *
from utils.utils import *

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image

stop_sign_area_thres = 30000

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

    # Run inference
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
            p, im0 = path[i], im0s[i].copy()

            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, _ in reversed(det):
                    area = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))

                    #yellow (0, 255, 255)
                    # red (0, 0, 255)
                    # green (0, 255, 0)

                    print(area)

                    if area < stop_sign_area_thres:
                        color = (0, 0, 255)
                    elif area == stop_sign_area_thres:
                        color = (0, 255, 255)
                    else: 
                        color = (0, 255, 0)

                    plot_one_box(xyxy, im0, label='stop sign %.2f' % (conf), color=color)

            # Print time (inference + NMS)
            #print('%sDone. (%.3f FPS)' % (s, 1 / (t2 - t1)))
            cv2.imshow(p, im0)

            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='conf/yolov4-custom-for-torch.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='weights/yolov4-custom-for-torch_best.pt', help='weights path')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    print(opt)

    with torch.no_grad():
        detect()
