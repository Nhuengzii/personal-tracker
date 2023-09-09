import cv2
from cv2.typing import MatLike
import argparse

from src.detectors import BaseDetector
from src.embedders import BaseEmbedder
from src.personal_trackers import BasePersonalTracker
from src.extentions.hand_triger import HandTriggerResult

def main(source: str | int, args):
    cap = cv2.VideoCapture(source)

    detector = BaseDetector()
    embedder = BaseEmbedder()
    tracker = BasePersonalTracker(detector, embedder, auto_add_target_features=args.auto_add_target_features, auto_add_target_features_interval=args.auto_add_target_features_interval)
    hand_trigger = HandTriggerResult()

    def get_target_from_camera(cap) -> tuple[MatLike, tuple[int, int, int, int]]:
        target_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). continue ...")
                continue
            cv2.imshow("target", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                target_frame = frame
                break
        soi = cv2.selectROI("target", target_frame) 
        # convert to x1, y1, x2, y2
        soi = (int(soi[0]), int(soi[1]), int(soi[0]) + int(soi[2]), int(soi[1]) + int(soi[3]))
        cv2.destroyAllWindows()
        return target_frame, soi

    target_frame, soi = get_target_from_camera(cap)
    tracker.add_target_features(target_frame, soi)
    print(f"Target added current size {len(tracker._target_features_pool)}")
    target_frame, soi = get_target_from_camera(cap)
    tracker.add_target_features(target_frame, soi)
    print(f"Target added current size {len(tracker._target_features_pool)}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        track_result = tracker.track(frame, draw_result=True)
        cv2.imshow("target", tracker.show_target_images())
        
        if track_result:
            crop_target =  frame[
                track_result.detect_result.bboxes[0][1]:track_result.detect_result.bboxes[0][3],
                track_result.detect_result.bboxes[0][0]:track_result.detect_result.bboxes[0][2]
                ]
            cv2.imshow("crop_target", crop_target)
        hand_target, is_stop = hand_trigger.find_hands(crop_target)
        if is_stop:
            print("Stop")
        else:
            print("Not Stop")
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=0, help="source of video path or camera index")
    parser.add_argument("--auto_add_target_features", type=bool, default=False, help="auto add target features")
    parser.add_argument("--auto_add_target_features_interval", type=int, default=60, help="auto add target features interval")
    args = parser.parse_args()
    if args.source.isdigit():
        vid_source = int(args.source)
    else:
        vid_source = args.source
    main(source=vid_source, args=args)
    
