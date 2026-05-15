from multiprocessing import Process, Queue
from detector import run_detector
from controller import run_controller


def main():
    queue = Queue(maxsize=1)

    detector_process = Process(target=run_detector, args=(queue,))
    controller_process = Process(target=run_controller, args=(queue,))

    detector_process.start()
    controller_process.start()

    detector_process.join()
    controller_process.join()


if __name__ == "__main__":
    main()
