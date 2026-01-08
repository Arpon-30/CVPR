import cv2
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "face_classifier_mobilenetv2.keras"
CLASS_PATH = "class_names.json"
THRESHOLD  = 0.70

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

def preprocess_face(roi_bgr):
    roi = cv2.resize(roi_bgr, (224,224))
    roi = roi.astype("float32")
    roi = preprocess_input(roi)

    x = np.expand_dims(roi, axis=0)  # (1,224,224,3)
    return x, roi


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'q' to quit")

BOX_SIZE = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    x1 = w//2 - BOX_SIZE//2
    y1 = h//2 - BOX_SIZE//2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    roi = frame[y1:y2, x1:x2]

    x_inp, roi_proc = preprocess_face(roi)

    probs = model.predict(x_inp, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    if conf >= THRESHOLD:
        label = f"{class_names[pred_id]} ({conf:.2f})"
        color = (0,255,0)
    else:
        label = f"Unknown ({conf:.2f})"
        color = (0,0,255)

    # Draw ROI box
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)

    # Model input preview (MNIST-style)
    preview = roi_proc.copy()
    preview = (preview - preview.min()) / (preview.max() - preview.min())
    preview = (preview * 255).astype(np.uint8)
    preview = cv2.resize(preview, (140,140))
    frame[10:150, w-150:w-10] = preview

    cv2.putText(frame, "Model Input",
                (w-150, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 1)

    cv2.imshow("Face Recognition (MobileNetV2)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
