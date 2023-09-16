
import cv2
from personal_tracker.helpers import draw_bbox, crop_image_from_bbox
from personal_tracker.extentions.hand_triger import HandTriggerResult
from personal_tracker.detector import Detector
import math
from cv2.typing import MatLike, VideoCapture

hand_trigger = HandTriggerResult()
min_diff = 100000
target_person = None

cap = cv2.VideoCapture(0)

def find_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_target_with_hand(cap):
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detector_result = Detector().detect(frame)
        if detector_result.bboxes is None:
            continue
                
        for bbox in detector_result.bboxes:
            center_person = hand_trigger.get_center_of_frame(bbox)
            frame_person = crop_image_from_bbox(frame, bbox)
            hand_person, hand_lm = hand_trigger.process_hands(frame_person,True)
            if not hand_trigger.is_detect:
                target_person = None
                continue
            #cv2.imshow("target", frame_person)
            if target_person is None:
                target_person = hand_trigger.get_center_of_frame(bbox)
                        
            cv2.putText(frame, "*", target_person, cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5, cv2.LINE_AA)
            distance = find_distance(target_person, center_person)
            #diff = math.sqrt((target_person[0] - center_person[0])**2 + (target_person[1] - center_person[1])**2)
            distance_text = f"Diff: {distance:.2f}"
            cv2.putText(frame, distance_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            if distance < min_diff:
                min_diff = distance
                target_bbox = bbox
                target_img = crop_image_from_bbox(frame, target_bbox)
            cv2.imshow("target", target_img)
            if target_person and hand_trigger.is_detect:
                return target_img
                pass
                    #start detect

cap = cv2.VideoCapture(0)
target_img = find_target_with_hand(cap)
cv2.imshow("target", target_img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    cap.release()
    exit()
