from ultralytics import YOLO
import cv2


model = YOLO("best.torchscript", task="detect")  # load a custom model
camera = cv2.VideoCapture(0)
confidence_threshold = 0.5

assert camera.isOpened()

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        #x = 0.32
        #frame = frame[:, round(1280 * x):round(1280 * (1 - x))].transpose(1, 0, 2)

        # Inference
        predictions = model(frame, imgsz=[256, 320])

        for prediction in predictions:
            # Extract all the box coords, class ids, and confidences
            xyxy = prediction.boxes.xyxy.cpu().numpy()  # shape: [num_boxes, 4]
            cls_ids = prediction.boxes.cls.int().cpu().numpy()
            confs = prediction.boxes.conf.cpu().numpy()

            # Loop through detections and draw
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                if conf < confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                label = f"{prediction.names[cls_id]} {conf:.2f}"

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                # Draw filled box behind text for readability
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - int(1.2 * h)), (x1 + w, y1), (0, 255, 0), -1)

                # Put the class+conf text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

        # Display the image in a window
        cv2.imshow('YOLOv11 Detections', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()