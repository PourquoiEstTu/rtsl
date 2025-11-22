import cv2
import numpy as np
import torch
import mediapipe as mp
import time

# IMPORT MODEL
from model import GRU_1DCNN, CNN_BiGRU_Attention  # you used this model in training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# code to intialize count down for starting prediction
countdown_active = True
countdown_start = 0
countdown_time = 3

# MATCH THE TRAINING SETTINGS
INPUT_SIZE = 126
WINDOW_SIZE = 60 # number of frames per prediction
PATH = ""
MODEL_PATH = f"{PATH}trainingNN.pth"

# LOAD LABEL ENCODER CLASSES
classes = np.load(f"{PATH}classes.npy")   # array of strings
print(f"Loaded {len(classes)} classes: {classes}")

# LOAD NORMALIZATION STATISTICS
mean = np.load(f"{PATH}mean.npy")   # shape (1,1,126)
std  = np.load(f"{PATH}std.npy")

# LOAD TRAINED MODEL
num_classes = len(classes)

model = CNN_BiGRU_Attention(input_size=INPUT_SIZE, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("loaded model")

# PREDICT FUNCTION
def predict(seq):
    seq = (seq - mean) / std  
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x = x.reshape(1, WINDOW_SIZE, INPUT_SIZE)

    with torch.no_grad():
        logits = model(x)
        idx = torch.argmax(logits, dim=1).item()

    return classes[idx]


# 6) MEDIAPIPE SETUP
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

buffer = []
current_pred = ""



# 7) LIVE LOOP
prev_time = time.time()
fps = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flip camera
    frame = cv2.flip(frame, 1)

    # fps calculation
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
    prev_time = curr_time

    key = cv2.waitKey(1) & 0xFF

    # start countdown on 's'
    if key == ord('s') and not countdown_active:
        countdown_active = True
        countdown_start = time.time()

    # show countdown
    if countdown_active:
        elapsed = time.time() - countdown_start
        remaining = int(countdown_time - elapsed) + 1
        if remaining > 0:
            cv2.putText(frame, f"Starting in {remaining}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.imshow("Live Prediction", frame)
            continue
        else:
            countdown_active = False
            buffer = []
            current_pred = ""
            print("Countdown complete → Starting prediction!")

    # process with mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            h, w, _ = frame.shape
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # pad if needed
    if len(landmarks) < INPUT_SIZE:
        landmarks += [0.0] * (INPUT_SIZE - len(landmarks))
    elif len(landmarks) > INPUT_SIZE:
        landmarks = landmarks[:INPUT_SIZE]

    landmarks = np.array(landmarks)
    buffer.append(landmarks)

    # Only predict if **hands are present**
    if np.sum(landmarks) != 0 and len(buffer) >= WINDOW_SIZE:
        seq = np.stack(buffer[-WINDOW_SIZE:])
        current_pred = predict(seq)
    elif np.sum(landmarks) == 0:
        current_pred = ""  # no hands detected

    # Draw prediction at bottom-center
    (h, w, _) = frame.shape
    text = current_pred

    # Get text size for perfect centering
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,   # scale
        4      # thickness
    )

    # Bottom center position
    x = (w - text_w) // 2
    y = h - 40   # 40px above bottom edge

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        4
    )

    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Live Prediction", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()