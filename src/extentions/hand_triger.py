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
        self.mcplist = [5, 9, 13, 17]
        self.total_fingers = 0
        self.flag_start_stop = False
        self.time_start = 0
        self.is_stop = False
        self.results = None
        self.time_floded_down_list = []
        self.count_floded_down = 0
        self.is_detect = False

    def _has_results(self):
        return self.results.multi_hand_landmarks is not None

    def _check_stop(self, time_to_stop=4):
        if self.total_fingers >= 4:
            if not self.flag_start_stop:
                print("Start condition met: Starting process.")
                self.time_start = time.time()
                self.flag_start_stop = True
            elif time.time() - self.time_start >= time_to_stop:
                self.is_stop = not self.is_stop
                self.flag_start_stop = False
        elif self.total_fingers < 4 and self.flag_start_stop:
            print("Stop condition met: Stopping process.")
            self.flag_start_stop = False

    def is_beckon(self):
        if self._has_results():
            for hand_lms in self.results.multi_hand_landmarks:
                fingers = [1 if hand_lms.landmark[self.tiplist[0]].x < hand_lms.landmark[self.mcplist[0]] else 0]

                for id in range(1, 5):
                    fingers.append(1 if hand_lms.landmark[self.tiplist[id]].y < hand_lms.landmark[self.mcplist[id - 1]] else 0)

                total_fingers = fingers.count(1)

                if total_fingers <= 2:
                    duration_finger_folded_down = 0
                    time_finger_floded_down = time.time()
                    self.time_floded_down_list.append(time_finger_floded_down)
                else:
                    if len(self.time_floded_down_list) > 0:
                        duration_finger_folded_down = self.time_floded_down_list[-1] - self.time_floded_down_list[0]
                        self.time_floded_down_list.clear()

                        if duration_finger_folded_down >= 0.12:
                            self.count_floded_down += 1
                            print("count_floded_down:", self.count_floded_down)

                if self.count_floded_down >= 2:
                    print("Beckon")
                    self.count_floded_down = 0
                    return True
                    

    def find_command_detecting(self, cv_img: MatLike):
        
        img_RGB = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        results = self.hands.process(img_RGB)
        total_fingers = []
        fingers = []
        fingers_stop = []
        if results.multi_hand_landmarks is None:
            return False
        
        else:
            for hand_lms in results.multi_hand_landmarks:
                if hand_lms.landmark[4].y > hand_lms.landmark[0].y and self.is_detect:
                    self.is_detect = False

                fingers = [1 if hand_lms.landmark[self.tiplist[0]].y > hand_lms.landmark[self.mcplist[0]].y else 0]
                for id in range(1, 5):
                    if id == 2 or id == 3:
                        fingers.append(1 if hand_lms.landmark[self.tiplist[id]].y > hand_lms.landmark[self.mcplist[id - 1]].y else 0)
                    else:
                        fingers.append(1 if hand_lms.landmark[self.tiplist[id]].y < hand_lms.landmark[self.mcplist[id - 1]].y else 0)
                        
                total_fingers = fingers.count(1)
                if total_fingers >= 5:
                    self.is_detect = True
                
        return self.is_detect
    
    def show_status(self, cv_img: MatLike):
        if self.is_detect:
            cv.putText(cv_img, "start Detecting", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            if self.is_stop:
                cv.putText(cv_img, "stop", (50, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
            else:
                cv.putText(cv_img, "start", (50, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
        else:
            cv.putText(cv_img, "stop detect", (50, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
        
    def find_hands(self, cv_img: MatLike, draw=True):
        img_RGB = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        w, h =cv_img.shape[:2]
        fingers = []

        if self._has_results() and self.is_detect:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(cv_img, hand_lms, self.mp_hand.HAND_CONNECTIONS)

                if hand_lms.landmark[self.tiplist[0]].x < hand_lms.landmark[self.tiplist[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if hand_lms.landmark[self.tiplist[id]].y < hand_lms.landmark[self.tiplist[id] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                self.total_fingers = fingers.count(1)

                self._check_stop()
        else:
            cv.putText(cv_img, "No hand detected", (w-200, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            self.flag_start_stop = False
            self.time_start = 0
        return cv_img, self.is_stop
