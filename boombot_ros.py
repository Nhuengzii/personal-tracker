from personal_tracker import PersonalTracker, TrackResult 
from personal_tracker.extentions.hand_triger import HandTriggerResult
import cv2
from datetime import datetime

from personal_tracker.tracker.tracker_config import TrackerConfig

                
def main(cap):
    # Setup Tracker
    config = TrackerConfig()
    tracker = PersonalTracker(config)
    
    def find_target(cap) -> tuple[int, int, int, int]:
        hand_trigger = HandTriggerResult()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            detect_result = tracker._detector.detect(frame)
            if detect_result is None:
                continue

            for target in detect_result.bboxes:
                croped = frame[target[1]:target[3], target[0]:target[2]]
                hand_trigger.process_hands(croped, draw=True)
                if hand_trigger.is_detect:
                    cv2.destroyWindow("result")
                    return target
            cv2.waitKey(1)
            cv2.imshow("result", frame)
    def get_target(cap, target_bbox, second=3):
        cx, cy = (target_bbox[0] + target_bbox[2]) // 2, (target_bbox[1] + target_bbox[3]) // 2
        targets = []
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < second:
            ret, frame = cap.read()
            if not ret:
                continue

            detect_result = tracker._detector.detect(frame)
            if detect_result is None:
                continue
            ranks = tracker._tracker._kalman_ranking((cx, cy), detect_result.bboxes)
            target_bbox = detect_result.bboxes[ranks[0]]
            targets.append((frame, target_bbox))
            cv2.waitKey(200)
        return targets
        

    target_bbox = find_target(cap)
    targets = get_target(cap, target_bbox)
    for target in targets:
        tracker.add_target_features(target[0], target[1])
    print(f"Start tracking with {len(targets)} num targets")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result = tracker.track(frame, draw_result=True)
        if not result.success:
            continue
        
        if cv2.waitKey(1) == ord("q"):
            break
        cv2.imshow("result", frame)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    main(cap)
