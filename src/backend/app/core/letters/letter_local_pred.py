import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from collections import deque
import mediapipe as mp

import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ── Settings ─────────────────────────────
MODEL_PATH = "saved_models/mobilenetv2_best_phase2.pth"

class_names = [
'A','B','C','D','E','F','G','H','I','J','K','L','M',
'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'del','nothing','space'
]

BUFFER_SIZE = 8
CONF_THRESHOLD = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ───────────────────────────
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model = model.to(device)
model.eval()

# ── Image transform ──────────────────────
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ── MediaPipe Hands ──────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ── Buffers ──────────────────────────────
prediction_buffer = deque(maxlen=BUFFER_SIZE)
sentence_buffer = []
last_letter = ""
stable_count = 0

# ── Predict function ─────────────────────
def predict(hand_img):

    img = Image.fromarray(hand_img)

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()

# ── Smooth prediction ────────────────────
def smooth_prediction():

    if len(prediction_buffer) == 0:
        return ""

    counts = {}

    for p in prediction_buffer:
        counts[p] = counts.get(p,0) + 1

    return max(counts, key=counts.get)

# ── Webcam ───────────────────────────────
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            xs = []
            ys = []

            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            x_min = max(min(xs) - 20, 0)
            x_max = min(max(xs) + 20, w)
            y_min = max(min(ys) - 20, 0)
            y_max = min(max(ys) + 20, h)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size == 0:
                continue

            letter, conf = predict(hand_crop)

            if conf > CONF_THRESHOLD:
                prediction_buffer.append(letter)

            smooth_letter = smooth_prediction()

            prediction_text = f"{smooth_letter} ({conf:.2f})"

            # ── Sentence buffer logic ──
            if smooth_letter == last_letter:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count > 6:
                if len(sentence_buffer) == 0 or sentence_buffer[-1] != smooth_letter:

                    if smooth_letter == "space":
                        sentence_buffer.append(" ")

                    elif smooth_letter == "del":
                        if len(sentence_buffer) > 0:
                            sentence_buffer.pop()

                    elif smooth_letter != "nothing":
                        sentence_buffer.append(smooth_letter)

            last_letter = smooth_letter

            cv2.rectangle(
                frame,
                (x_min,y_min),
                (x_max,y_max),
                (0,255,0),
                2
            )

    # ── Draw predictions ──────────────────
    cv2.putText(
        frame,
        prediction_text,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    sentence = "".join(sentence_buffer)

    cv2.putText(
        frame,
        sentence,
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    cv2.imshow("ASL Live Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()