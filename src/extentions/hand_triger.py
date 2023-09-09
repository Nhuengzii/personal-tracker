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
        self.flag_start_stop = False
        self.time_start = 0
        self.is_stop = False
        self.results = None
    
    def _check_stop(self, time_to_stop=4) -> bool:
            duration_time = time.time() - self.time_start
            #print(duration_time)
            if self.total_fingers >= 4:
                #print(time.time() - self.time_start)
                #print("is finger >= 4")
                if not self.flag_start_stop:
                    print("Start condition met: Starting process.")
                    self.time_start = time.time()
                    self.flag_start_stop = True
                

                elif duration_time >= time_to_stop:
                    #print("Stop condition met: Stopping process.")
                    self.is_stop = not self.is_stop
                    self.flag_start_stop = False
                    duration_time = time.time() - self.time_start
                    

            elif (self.total_fingers < 4 and self.flag_start_stop):
                print("Stop condition met: Stopping process.")    
                self.flag_start_stop = False
                duration_time = 0
                
            
            return self.is_stop
    
    def _comeon_sign(self):
        pass

    def find_hands(self, cv_img: MatLike, draw=True):
        img_RGB = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        fingers = []

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
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
                self._check_stop()
        else:
            self.flag_start_stop = False
            self.time_start = 0
        return cv_img, self.is_stop
    
    
    
    def update_is_stop(self):
        return self.is_stop
    

