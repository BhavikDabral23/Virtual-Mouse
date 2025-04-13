import cv2
import mediapipe as mp
import pyautogui
import psutil
import time
from pynput.mouse import Button, Controller

mouse = Controller()
screen_width, screen_height = pyautogui.size()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)

def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        index_tip = landmark_list[mpHands.HandLandmark.INDEX_FINGER_TIP]
        index_base = landmark_list[mpHands.HandLandmark.INDEX_FINGER_PIP]
        pinky_tip = landmark_list[mpHands.HandLandmark.PINKY_TIP]
        pinky_base = landmark_list[mpHands.HandLandmark.PINKY_PIP]
        middle_tip = landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_base = landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_PIP]

        index_distance = calculate_distance(index_tip, index_base)
        pinky_distance = calculate_distance(pinky_tip, pinky_base)
        middle_distance = calculate_distance(middle_tip, middle_base)

        threshold = 0.04

        if pinky_distance < threshold:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif index_distance < threshold:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif middle_distance < threshold:
            pyautogui.scroll(-5)
            cv2.putText(frame, "Scrolling", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        index_finger_tip = find_finger_tip(processed)
        if index_finger_tip:
            move_mouse(index_finger_tip)

def process_frame(frame, landmark_list, processed):
    detect_gesture(frame, landmark_list, processed)

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        landmark_list = []
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.append((lm.x, lm.y))

        process_frame(frame, landmark_list, processed)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
