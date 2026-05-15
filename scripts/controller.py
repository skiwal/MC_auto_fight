import time
import pyautogui

def run_controller(queue):
    while True:
        target = queue.get()

        if not target["found"]:
            print("No target")
            continue

        center_x = target["center_x"]
        center_y = target["center_y"]

        screen_x = target["screen_x"]
        screen_y = target["screen_y"]

        x1 = target["x1"]
        y1 = target["y1"]
        x2 = target["x2"]
        y2 = target["y2"]

        conf = target["conf"]

        print(
            f"Target found: center=({center_x}, {center_y}), "
            f"screen=({screen_x}, {screen_y}), conf={conf:.2f}"
        )

        # 示例：移动鼠标到目标中心

        pyautogui.moveTo(screen_x, screen_y)
        while pyautogui.position() > (x1, y2) and pyautogui.position() < (x2, y1):
            pyautogui.click()
        time.sleep(0.01)
