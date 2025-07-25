import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import math


bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = green_index = red_index = yellow_index = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),(255,255,255)]
colorIndex = 0

paintWindow = np.ones((1000, 900, 3), dtype=np.uint8) * 255
cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Paint', 1000, 600)

#def draw_buttons(win):
#     cv2.rectangle(win, (40, 1), (140, 65), (0, 0, 0), 2)
#     cv2.rectangle(win, (160, 1), (280, 65), (255, 0, 0), 2)
#     cv2.rectangle(win, (300, 1), (420, 65), (0, 255, 0), 2)
#     cv2.rectangle(win, (440, 1), (560, 65), (0, 0, 255), 2)
#     cv2.rectangle(win, (580, 1), (700, 65), (0, 255, 255), 2)
#     cv2.rectangle(win, (720, 1), (840, 65), (128, 128, 128), 2)  # SAVE
#
#     cv2.putText(win, "CLEAR", (49, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     cv2.putText(win, "BLUE", (185, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     cv2.putText(win, "GREEN", (320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     cv2.putText(win, "RED", (470, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     cv2.putText(win, "YELLOW", (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     cv2.putText(win, "SAVE", (750, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
def draw_buttons(win):
    button_width = 100
    button_height = 64
    spacing = 20
    start_x = 20

    labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLOW", "SAVE"]
    colors_border = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (128, 128, 128)]

    for i, label in enumerate(labels):
        x1 = start_x + i * (button_width + spacing)
        x2 = x1 + button_width
        y1, y2 = 1, 65
        cv2.rectangle(win, (x1, y1), (x2, y2), colors_border[i], 2)
        cv2.putText(win, label, (x1 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 1200, 800)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    draw_buttons(frame)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in handslms.landmark]
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = tuple(landmarks[8])
        thumb = tuple(landmarks[4])
        cv2.circle(frame, fore_finger, 5, (0, 255, 0), -1)

        # if abs(fore_finger[1] - thumb[1]) < 40:
        #     bpoints.append(deque(maxlen=1024)); blue_index += 1
        #     gpoints.append(deque(maxlen=1024)); green_index += 1
        #     rpoints.append(deque(maxlen=1024)); red_index += 1
        #     ypoints.append(deque(maxlen=1024)); yellow_index += 1
        distance = math.hypot(fore_finger[0] - thumb[0], fore_finger[1] - thumb[1])
        if distance < 40:
            bpoints.append(deque(maxlen=1024)); blue_index += 1
            gpoints.append(deque(maxlen=1024)); green_index += 1
            rpoints.append(deque(maxlen=1024)); red_index += 1
            ypoints.append(deque(maxlen=1024)); yellow_index += 1


        elif fore_finger[1] <= 65:
            x = fore_finger[0]
            if 20 <= x <= 120:
                bpoints = [deque(maxlen=1024)]; blue_index = 0
                gpoints = [deque(maxlen=1024)]; green_index = 0
                rpoints = [deque(maxlen=1024)]; red_index = 0
                ypoints = [deque(maxlen=1024)]; yellow_index = 0
                paintWindow[70:, :] = 255
            elif 140 <= x <= 240: colorIndex = 0
            elif 260 <= x <= 360: colorIndex = 1
            elif 380 <= x <= 480: colorIndex = 2
            elif 500 <= x <= 600: colorIndex = 3
            elif 620 <= x <= 720:

                filename = f"drawing_{int(time.time())}.png"
                cv2.imwrite(filename, paintWindow)
                cv2.putText(frame, f"Saved as {filename}", (300, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(fore_finger)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(fore_finger)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(fore_finger)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(fore_finger)
    else:
        bpoints.append(deque(maxlen=1024)); blue_index += 1
        gpoints.append(deque(maxlen=1024)); green_index += 1
        rpoints.append(deque(maxlen=1024)); red_index += 1
        ypoints.append(deque(maxlen=1024)); yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 5)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 5)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

