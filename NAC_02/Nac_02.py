# -- coding: utf-8 --

from ast import match_case
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("q1.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #Meu RM81942 = as de ouros

    template = cv2.imread("aOuros.png", 0)
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(imgGray, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    larg, alt = template.shape[::-1]
    bottom_right = (min_loc[0] + larg, min_loc[1] + alt)
    cv2.rectangle(frame, min_loc, bottom_right, (0, 255, 0), 3)
    cv2.putText(frame, 'CARTA DETECTADA', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(200,50,0),2,cv2.LINE_AA)

    # Exibe resultado
    cv2.imshow("Feed", frame)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()