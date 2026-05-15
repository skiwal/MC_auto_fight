import time
import cv2
import numpy as np
import mss
from ultralytics import YOLO


def put_latest(queue, data):
    """
    队列只保留最新的一条数据，避免控制端处理旧坐标。
    """
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break

    queue.put(data)


def run_detector(queue):
    model = YOLO("runs/mc_yolo26/mc_mob_yolo26n/weights/best.pt")

    monitor = sct.monitors[1]

    with mss.mss() as sct:
        while True:
            screenshot = sct.grab(monitor)

            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model(frame, conf=0.5, verbose=False)
            result = results[0]

            best_target = None
            best_conf = 0.0

            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if conf > best_conf:
                    best_conf = conf
                    best_target = {
                        "found": True,
                        "class_id": cls_id,
                        "conf": conf,

                        # 相对于截图区域的坐标
                        "x1" : int(x1),
                        "y1" : int(y1),
                        "x2" : int(x2),
                        "y2" : int(y2),
                        "center_x": center_x,
                        "center_y": center_y,

                        # 相对于整个屏幕的绝对坐标
                        "screen_x": monitor["left"] + center_x,
                        "screen_y": monitor["top"] + center_y,

                        "time": time.time()
                    }

            if best_target is None:
                best_target = {
                    "found": False,
                    "time": time.time()
                }

            put_latest(queue, best_target)
