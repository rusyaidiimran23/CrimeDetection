import cv2, time
from ultralytics import YOLO


SENSOR_ID = 0
model = YOLO("crimeV4.pt")
names = model.names

# Set your class thresholds here (by class name)
CLASS_THRES = {
    "person": 0.30,
    "phone": 0.40,
    "knife": 0.15,
    "gun": 0.65,
    "mask": 0.30,
}

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    out = frame

    # run detection every 3 frames (tune 2/3/4)
    if frame_id % 2 == 0:
        r = model.predict(frame, imgsz=320, conf=0.20, verbose=False)[0]
        
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
        
            th= CLASS_THRES.get(cls_name, 0.50)
        
            if conf < th:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(out, (int(x1), int(y1), int(x2), int(y2)), (0,255,0),2)
            cv2.putText(out, f"{cls_name} {conf:.2f}",
                (int(x1), max(0, int(y1)-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2) 
     

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
