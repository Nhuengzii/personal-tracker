import cv2
from cv2.typing import MatLike
import argparse
from src.detectors import BaseDetector
from src.detectors.segment_detector import SegmentDetector
from src.embedders import BaseEmbedder
from src.embedders.available_embedder_models import AvailableEmbedderModels
from src.metrics.base_metric import MetricType
import os
from src.personal_trackers.me_personal_tracker import MEPersonalTracker
from src.personal_trackers.personal_tracker import PersonalTracker
from src.personal_trackers.sift_personal_tracker import SIFTPersonalTracker


def main(source: str | int, args):
    print(f"Source: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = BaseDetector()
    embedder1 = BaseEmbedder(model=AvailableEmbedderModels.OSNET_AIN_X1_0)
    embedder2 = BaseEmbedder(model=AvailableEmbedderModels.OSNET_AIN_X1_0)
    # embedders = [BaseEmbedder(AvailableEmbedderModels.OSNET_X1_0), BaseEmbedder(AvailableEmbedderModels.OSNET_AIN_X1_0)]
    # embedders.append(BaseEmbedder(AvailableEmbedderModels.OSNET_X0_75))
    # embedders.append(BaseEmbedder(AvailableEmbedderModels.OSNET_AIN_X0_75))
    tracker1 = SIFTPersonalTracker(SegmentDetector(), embedder1, MetricType.COSINE_SIMILARITY)
    # tracker1 = PersonalTracker(detector, embedder1, MetricType.CSEM_DISTANCE) 
    tracker2 = PersonalTracker(detector, embedder2, MetricType.CSEM_DISTANCE)
    cur = os.getcwd()
    targets = tracker1.get_target_from_camera(cap, 8)
    for target in targets:
        tracker1.add_target_features(target[0], target[1])
        tracker2.add_target_features(target[0], target[1])
    # for target in glob.glob(f"{cur}/validates/validate_sets/{args.set}/target_images/*.png"):
    #     print(target)
    #     croped_image = cv2.imread(target)
    #     full_bbox = (0, 0, croped_image.shape[1], croped_image.shape[0])
    #     tracker1.add_target_features(croped_image, full_bbox)
    #     tracker2.add_target_features(croped_image, full_bbox)
    print(f"Tracker1 Start tracking with {len(tracker1._target_features_pool)} targets")
    print(f"Tracker2 Start tracking with {len(tracker2._target_features_pool)} targets")
    
    
    while True:
        ret, frame = cap.read()
        frame1 = frame.copy()
        frame2 = frame.copy()
        if not ret:
            break
        track_result = tracker1.track(frame1, draw_result=True)
        track_result = tracker2.track(frame2, draw_result=True)
        concat_frame = cv2.hconcat([frame1, frame2])
        concat_frame = cv2.resize(concat_frame, (1280, 720))
        cv2.imshow("target", tracker2.show_target_images())
        cv2.imshow('frame', concat_frame)
        if cv2.waitKey(1) == ord(' '):
            while True:
                if cv2.waitKey(1) == ord(' '):
                    break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Start validate")
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='set1', help='validate set')
    
    args = parser.parse_args()
    cur = os.getcwd()
    main(f"{cur}/validates/validate_sets/{args.set}/source.webm", args)
