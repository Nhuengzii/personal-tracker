import cv2 as cv
import mediapipe as mp
import time
from cv2.typing import MatLike

class HandTriggerResult:
    def __init__(self):
        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.tiplist = [4, 8, 12, 16, 20]
        self.total_fingers = 0
        self.flag_is_start = False
        self.time_start = 0

    def _find_hands(self, cv_img: MatLike, draw=True):
        img_RGB = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        results = self.hands.process(img_RGB)
        fingers = []

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(cv_img, hand_lms, self.mp_hand.HAND_CONNECTIONS)
                if hand_lms.landmark[self.tiplist[0]].x < hand_lms.landmark[self.tiplist[0]-1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if hand_lms.landmark[self.tiplist[id]].y < hand_lms.landmark[self.tiplist[id]-2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                self.total_fingers = fingers.count(1)
        return cv_img

    # not working
    def check_update_stop(self) -> bool:
        if self.total_fingers >= 4:
            if not self.flag_is_start:
                self.time_start = time.time()
                self.flag_is_start = True
            elif time.time() - self.time_start > 5:
                self.flag_is_start = False
                return True
        else:
            self.flag_is_start = False

        return False

my_hand = HandTriggerResult()
cap = cv.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     img = my_hand._find_hands(img)
#     check = my_hand.check_update_stop()

#     cv.imshow('img', img)

#     if cv.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()
