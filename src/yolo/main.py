import cv2
from jetcam.csi_camera import CSICamera

camera = CSICamera(width=224, height=224, capture_width=1080, capture_height=720, capture_fps=30)

while True:
    img = camera.read()

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break