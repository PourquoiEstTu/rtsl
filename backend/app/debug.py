# (this code creates debug images with hand landmarks drawn on them, chatgpt generated)
import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("videos/27171.mp4")
debug_dir = "debug_frames_visual"
os.makedirs(debug_dir, exist_ok=True)

with mp_hands.Hands(max_num_hands=2) as hands:
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save every 30th frame for debugging
        if frame_count % 30 == 0:
            cv2.imwrite(f"{debug_dir}/frame_{frame_count}.jpg", frame)

        frame_count += 1

cap.release()
print(f"Saved debug frames to {debug_dir}")
