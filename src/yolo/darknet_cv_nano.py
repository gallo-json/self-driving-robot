import cv2
import numpy as np
import time
from jetcam.csi_camera import CSICamera

camera = CSICamera(width=224, height=224)

net = cv2.dnn.readNet("stop-sign/yolov4-custom-colab_best.weights", "stop-sign/conf/yolov4-custom-colab.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = ["stop_sign"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = cv2.FONT_HERSHEY_PLAIN

prev_frame_time = 0
new_frame_time = 0

while True:
    frame = camera.read()

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
            cv2.putText(frame, label + " " + str(round(conf, 2)), (x, y + 30), font, 3, color, 3)
    
    new_frame_time = time.time()
    
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    # converting the fps into integer 
    #fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
  
    # puting the FPS count on the frame 

    cv2.putText(frame, fps, (10, 50), font, 2, (255, 0, 0), 3)

    cv2.imshow("Image", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break