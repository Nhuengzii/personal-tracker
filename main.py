import cv2
from cv2.typing import MatLike
import argparse
from playsound import playsound

from src.detectors import BaseDetector
from src.detectors.segment_detector import SegmentDetector
from src.embedders import BaseEmbedder
from src.embedders.available_embedder_models import AvailableEmbedderModels
from src.embedders.clip_embedder import CLIPEmbedder
from src.metrics.base_metric import MetricType
from src.personal_trackers import TrackResults, PersonalTracker
from src.personal_trackers.me_personal_tracker import MEPersonalTracker
from src.personal_trackers.sift_personal_tracker import SIFTPersonalTracker


def main(source: str | int, args):
    # rtmp_url = "http://192.168.137.218:8080/video"
    cap = cv2.VideoCapture(source)
    # cap = cv2.VideoCapture(rtmp_url)

    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    detector = BaseDetector()
    embedder = BaseEmbedder()
    hand_trigger = HandTriggerResult()
    # tracker = PersonalTracker(detector, embedder, MetricType.CSEM_DISTANCE, auto_add_target_features=args.auto_add_target_features, auto_add_target_features_interval=args.auto_add_target_features_interval)
    tracker = SIFTPersonalTracker(SegmentDetector(), embedder, MetricType.EUCLIDEAN_DISTANCE)
    tracker._kalman_trust_threshold = 4
    playsound("./cap.mp3")
    targets = tracker.get_repetitive_target_from_camera(cap, 5)
    for target in targets:
        tracker.add_target_features(target[0], target[1])

    cv2.imshow("target", tracker.show_target_images())
    print(f"Start tracking with {len(tracker._target_features_pool)} targets")
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
    main(source=vid_source, args=args)
