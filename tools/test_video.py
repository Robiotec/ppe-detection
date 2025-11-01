import cv2
from ultralytics import YOLO

def main():
    model_path = "/home/robiotec/Documents/ppe-detection/src/PPE_yolo11n/weights/best.pt"
    video_path = "/home/robiotec/Documents/ppe-detection/src/test.mp4"  

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = video_path.rsplit('.', 1)[0] + '_detected.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Reproduciendo video...")
    print(f"El video se guardará en: {output_path}")
    window_name = "Resultados de Detección de PPE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer el frame.")
                break

            results = model.predict(frame, conf=0.5)
            result_image = results[0].plot(conf=False)

            out.write(result_image)

            cv2.imshow(window_name, result_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video guardado en: {output_path}")
        print("Terminando aplicación...")

if __name__ == "__main__":
    main()
