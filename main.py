import cv2
from ultralytics import YOLO
from scripts.camera_service import CameraService

def main():
    model_path = "/home/robiotec/Documents/ppe-detection/src/PPE_yolo11n/weights/best.pt"

    model = YOLO(model_path)

    camera_service = CameraService()
    print("Iniciando captura en vivo...")
    window_name = "Resultados"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    width, height = 1280, 720
    cv2.resizeWindow(window_name, width, height)
    try:
        while True:
            frame = camera_service.get_frame()
            if frame is None:
                continue

            results = model.predict(frame, conf=0.5)
            
            result_image = results[0].plot(conf=False)

            cv2.imshow("Resultados", result_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera_service.close()
        cv2.destroyAllWindows()
        print("Terminando aplicación...")

if __name__ == "__main__":
    main()
