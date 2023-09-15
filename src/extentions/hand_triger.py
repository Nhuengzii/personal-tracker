import cv2 as cv
import mediapipe as mp
import time
from cv2.typing import MatLike
from typing import Tuple
from personal_tracker.helpers import crop_image_from_bbox
from personal_tracker.kalman_filter import KalmanFilter


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
        self.results = None
        self.is_stop = False
        self.is_detect = False
        self.is_beckon = False
        self.time_floded_down_list = []
        self.count_floded_down = 0
        self.duration_time = 0

    def _has_results(self):
        return self.results.multi_hand_landmarks is not None
    
    def check_is_stop(self):
        return self.is_stop
    
    def check_is_detect(self):
        return self.is_detect
    
    def check_is_beckon(self):
        return self.is_beckon
    
    def first_person(self, result:list, cv_img:MatLike):
        for bbox in result:
            frame_person = crop_image_from_bbox(cv_img, bbox)
            #for lm in self.results.multi_hand_landmarks:   
            self.process_hands(frame_person, True)
            

    def find_beckon(self, hand_lms):
        fingers = []
        tip_x =  hand_lms.landmark[self.tiplist[0]].x
        mcp_x = hand_lms.landmark[self.mcplist[0]].x
        fingers = [1 if tip_x < mcp_x else 0]

        for id in range(1, 5):
            tip_y = hand_lms.landmark[self.tiplist[id]].x
            mcp_y = hand_lms.landmark[self.mcplist[id - 1]].x
            fingers.append(1 if tip_y < mcp_y else 0)
            
            total_fingers = fingers.count(1)

            if total_fingers <= 2:
                duration_finger_folded_down = 0
                time_finger_floded_down = time.time()
                self.time_floded_down_list.append(time_finger_floded_down)
            else:
                if len(self.time_floded_down_list) > 0:
                    duration_finger_folded_down = self.time_floded_down_list[-1] - self.time_floded_down_list[0]
                    self.time_floded_down_list.clear()

                if duration_finger_folded_down < 0.12:
                        continue
                
                self.count_floded_down += 1
                print("count_floded_down: ----------------------> ", self.count_floded_down)

            if self.count_floded_down >= 2:
                self.is_beckon = True
                self.count_floded_down = 0
                return self.is_beckon
            

    def get_center_of_frame(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        #filter = KalmanFilter()
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        center_coordinates = (center_x, center_y)
        return center_coordinates
             

    def is_hand_target(self, coord_center, width, height, hand_lm, wrist_point) -> bool:
        x = int(coord_center[0])
        y = int(coord_center[1])

        x_wrist = int(wrist_point[0] * width)
        y_wrist = int(wrist_point[1] * height)

        for hand_lms in self.results.multi_hand_landmarks:
            if hand_lms.landmark[self.tiplist[0]].x < hand_lms.landmark[self.mcplist[0]].x:
                if x_wrist < x:
                    return True
            else:
                if x_wrist > x:
                    return True

    
    
    def find_command_detecting(self, hand_lms):
        total_fingers = []
        fingers = []
        fingers_stop = []
        ip_list = [3,6,10,14,19]

        if hand_lms.landmark[4].y > hand_lms.landmark[0].y and self.is_detect:
            fingers_stop.append(1)

            for id in range(1, 5):
                tip_x = hand_lms.landmark[self.tiplist[id]].x
                ip_x = hand_lms.landmark[ip_list[id]].x
                fingers_stop.append(1 if tip_x < ip_x else 0)

        total_fingers_stop = fingers_stop.count(1)
        if total_fingers_stop >= 5:
            self.is_detect = False
            self.is_beckon = False
            

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

    def find_stop(self, hand_lms, time_to_stop=4,):
        fingers = []
        fingers = [1 if hand_lms.landmark[self.tiplist[0]].x < hand_lms.landmark[self.tiplist[0] - 1].x else 0] 
        for id in range(1, 5):
            if hand_lms.landmark[self.tiplist[id]].y < hand_lms.landmark[self.tiplist[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
            
        self.total_fingers = fingers.count(1)
        self.duration_time = time.time() - self.time_start
        if self.total_fingers >= 4:
            if not self.flag_start_stop:
                print("Start condition met: Starting process.")
                self.time_start = time.time()
                self.flag_start_stop = True
            elif self.duration_time >= time_to_stop:
                self.is_stop = not self.is_stop
                self.flag_start_stop = False
        elif self.total_fingers < 4 and self.flag_start_stop:
            print("Stop condition met: Stopping process.")
            self.flag_start_stop = False
            self.duration_time = 0
        
    def process_hands(self, cv_img: MatLike, draw=True):
        img_RGB = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        
        detected_hand_landmarks = []  # Initialize a list to store detected hand landmarks

        if not self._has_results():
            self.flag_start_stop = False
            self.time_start = 0
            self.duration_time = 0
            return cv_img, detected_hand_landmarks  # Return an empty list if no hand landmarks found

        if draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(cv_img, hand_lms, self.mp_hand.HAND_CONNECTIONS)
                detected_hand_landmarks.append(hand_lms)  # Add detected hand landmarks to the list
            
        for hand_lms in self.results.multi_hand_landmarks:
            is_detect = self.find_command_detecting(hand_lms)
            if not is_detect:
                continue

            self.find_stop(hand_lms)
            #self.find_beckon(hand_lms)

        return cv_img, detected_hand_landmarks
    
    
    def show_status(self, cv_img: MatLike):
        status_text = "start Detecting" if self.is_detect else "stop detect"
        status_color = (0, 255, 0) if self.is_detect else (0, 0, 255)
        status_position = (10, 70)

        cv.putText(cv_img, status_text, status_position, cv.FONT_HERSHEY_PLAIN, 2, status_color, 3)

        if self.is_detect:
            action_text = "stop" if self.is_stop else "start"
            action_color = (0, 0, 255) if self.is_stop else (0, 255, 0)
            action_position = (50, 150)

            cv.putText(cv_img, action_text, action_position, cv.FONT_HERSHEY_PLAIN, 2, action_color, 5)
        cv.putText(cv_img, "Beckon" if self.is_beckon else "Not beckon", (10, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    