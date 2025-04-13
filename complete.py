import cv2
import mediapipe as mp
import pyautogui
import util
import random
import threading
from pynput.mouse import Button, Controller

mouse = Controller()
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

screenshot_taken = False 

def find_finger_tip(processed, finger_index):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[finger_index]
    return None

def move_mouse(index_finger_tip, prev_x, prev_y):
    if index_finger_tip is not None:
        x = int((prev_x * 0.7 + index_finger_tip.x * 0.3) * screen_width) 
        y = int((prev_y * 0.7 + index_finger_tip.y * 0.3) * screen_height)
        pyautogui.moveTo(x, y, duration=0.05)
        return x, y
    return prev_x, prev_y

def take_screenshot():
    global screenshot_taken
    if not screenshot_taken:
        screenshot_taken = True
        screenshot = pyautogui.screenshot()
        filename = f'screenshot_{random.randint(1000,9999)}.png'
        screenshot.save(filename)

def is_finger_extended(landmark_list, finger_tip, finger_base):
    return landmark_list[finger_tip][1] < landmark_list[finger_base][1]

def is_fist_closed(landmark_list):
    return all(landmark_list[finger_tip][1] > landmark_list[finger_base][1] for finger_tip, finger_base in [(8, 6), (12, 10), (16, 14), (20, 18)])

def detect_gesture(frame, landmark_list, processed, prev_x, prev_y):
    global screenshot_taken
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed, mpHands.HandLandmark.INDEX_FINGER_TIP)
        prev_x, prev_y = move_mouse(index_finger_tip, prev_x, prev_y)

        thumb_extended = is_finger_extended(landmark_list, 4, 2)  
        ring_extended = is_finger_extended(landmark_list, 16, 14)
        pinky_extended = is_finger_extended(landmark_list, 20, 18)
        fist_closed = is_fist_closed(landmark_list)

        if thumb_extended:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif ring_extended and not pinky_extended:
            pyautogui.scroll(10)
            cv2.putText(frame, "Scroll Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif pinky_extended and not ring_extended:
            pyautogui.scroll(-10)
            cv2.putText(frame, "Scroll Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        elif fist_closed:
            threading.Thread(target=take_screenshot, daemon=True).start()
            cv2.putText(frame, "Screenshot Taken", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            screenshot_taken = False 
    return prev_x, prev_y

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    prev_x, prev_y = 0, 0 

    try:
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

            prev_x, prev_y = detect_gesture(frame, landmark_list, processed, prev_x, prev_y)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
