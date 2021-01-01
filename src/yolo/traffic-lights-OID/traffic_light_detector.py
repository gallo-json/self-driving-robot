import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

net = cv2.dnn.readNet("weights/yolov4-custom_best.weights", "conf/yolov4-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = ["Traffic light"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = cv2.FONT_HERSHEY_PLAIN

frame = cv2.imread('none.jpg')
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

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        conf = confidences[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(frame, label + " " + str(round(conf, 2)), (x, y + 30), font, 3, color, 1)
        print(label)
        print(conf)
        
#cv2.imshow("Image", frame)

cropped_img = original[y - 2:y + h + 2, x - 2:x + w + 2]

cv2.imwrite('none-cropped.jpg', cropped_img)


#cv2.imshow("Cropped", cropped_img)
# wait until any key is pressed
#cv2.waitKey()