import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

net = cv2.dnn.readNet("weights/yolov4-custom_best.weights", "conf/yolov4-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

red_color = (0, 0, 255)
yellow_color = (255, 255, 0)
green_color = (0, 255, 0)

font = cv2.FONT_HERSHEY_PLAIN

frame = cv2.imread('pictures/yellow.jpg')
original = np.copy(frame)

height, width, channels = frame.shape

blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

lower = np.array([0, 0, 185])
upper = np.array([179, 155, 255])

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        conf = confidences[i]

        cropped_img = original[y - 2:y + h + 2, x - 2:x + w + 2]
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower, upper)

        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # If we have at least one contour, look through each one and pick the biggest
        if len(contours)>0:
            largest = 0
            area = 0
            for i in range(len(contours)):
                # get the area of the ith contour
                temp_area = cv2.contourArea(contours[i])
                # if it is the biggest we have seen, keep it
                if temp_area > area:
                    area = temp_area
                    largest = i
            # Compute the coordinates of the center of the largest contour
            coordinates = cv2.moments(contours[largest])
        biggest_area = cv2.contourArea(contours[largest])
        target_y = int(coordinates['m01']/coordinates['m00'])

        areas = cropped_img.shape[0] / 3

        if target_y < areas:
            cv2.rectangle(frame, (x, y), (x + w, y + h), red_color, 2)
            print("red")
        elif target_y > areas and target_y < areas * 2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), yellow_color, 2)
            print("yellow")
        elif target_y > areas * 2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), green_color, 2)
            print("green")

        # cv2.putText(frame, label + " " + str(round(conf, 2)), (x, y + 30), font, 3, color, 1)
        print(conf)
        
cv2.imshow("Image", frame)

# wait until any key is pressed
cv2.waitKey()