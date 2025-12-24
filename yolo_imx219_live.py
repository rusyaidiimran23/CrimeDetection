import cv2, time
from ultralytics import YOLO

SENSOR_ID = 0
model = YOLO("crimeV3.pt")

gst = (
    f"nvarguscamerasrc sensor-id={SENSOR_ID} ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=15/1 ! "
    "nvvidconv ! video/x-raw,width=640,height=360,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink max-buffers=1 drop=true sync=false"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

frame_id = 0
t0 = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # run detection every 3 frames (tune 2/3/4)
    if frame_id % 2 == 0:
        r = model.predict(frame, imgsz=416, conf=0.20, verbose=False)[0]
        out = r.plot()
    else:
        out = frame

    # FPS calc every 30 frames (separate from inference)
    if frame_id % 30 == 0:
        fps = 30 / (time.time() - t0)
        t0 = time.time()
        print("FPS:", round(fps, 1))

    cv2.imshow("Crime Model (Jetson)", out)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
