import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
smoothening = 7
click_cooldown = 1  # seconds
last_click_time = 0

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

            lm = handLms.landmark
            finger = {
                'thumb': (int(lm[4].x * w), int(lm[4].y * h)),
                'index': (int(lm[8].x * w), int(lm[8].y * h)),
                'middle': (int(lm[12].x * w), int(lm[12].y * h)),
                'ring': (int(lm[16].x * w), int(lm[16].y * h)),
                'pinky': (int(lm[20].x * w), int(lm[20].y * h)),
            }

            avg_dist = sum([
                distance(finger['thumb'], finger['index']),
                distance(finger['thumb'], finger['middle']),
                distance(finger['thumb'], finger['ring']),
                distance(finger['thumb'], finger['pinky']),
            ]) / 4

            current_time = time.time()

            if avg_dist < 60:
                # Move mouse
                screen_x = int(lm[8].x * screen_w)
                screen_y = int(lm[8].y * screen_h)
                curr_x = prev_x + (screen_x - prev_x) // smoothening
                curr_y = prev_y + (screen_y - prev_y) // smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                cv2.putText(frame, "Mouse Mode", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Left click: thumb close to index tip
                if distance(finger['thumb'], finger['index']) < 40:
                    if current_time - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time
                        cv2.putText(frame, "Left Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                # Right click: thumb close to middle tip
                elif distance(finger['thumb'], finger['middle']) < 40:
                    if current_time - last_click_time > click_cooldown:
                        pyautogui.rightClick()
                        last_click_time = current_time
                        cv2.putText(frame, "Right Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display output
    cv2.imshow("Hand Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
