import cv2
import imageio
from ultralytics import YOLO

def main():
    model_path = "/home/robiotec/Documents/ppe-detection/src/PPE_yolo11n/weights/best.pt"
    video_path = "/home/robiotec/Documents/ppe-detection/src/video_test.mp4"

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30 

    output_path = video_path.rsplit('.', 1)[0] + "_detected.mp4"

    print("Procesando detecciones con YOLO...")

    window_name = "Detección de EPP"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer el frame.")
                break

            results = model.predict(frame, conf=0.5)
            result_image = results[0].plot(conf=False)

            rgb_frame = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

            cv2.imshow(window_name, result_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción manual detectada.")
    finally:
        cap.release()
        writer.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
