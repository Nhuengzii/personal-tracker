import cv2
from cv2.typing import MatLike
import argparse

from src.detectors import BaseDetector
from src.embedders import BaseEmbedder
from src.embedders.available_embedder_models import AvailableEmbedderModels
from src.embedders.biased_embedder import BiasedEmbedder
from src.metrics.base_metric import MetricType
from src.personal_trackers import BasePersonalTracker
from src.personal_trackers.kalman_personal_tracker import KalmanPersonalTracker
import glob
import os


def main(source: str | int, args):
    print(f"Source: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = BaseDetector()
    embedder1 = BiasedEmbedder(model=AvailableEmbedderModels.OSNET_AIN_X1_0)
    embedder2 = BaseEmbedder(model=AvailableEmbedderModels.OSNET_AIN_X1_0)
    tracker1 = KalmanPersonalTracker(detector, embedder1, MetricType.MAHALANOBIS_KALMAN_DISTANCE) 
    tracker2 = KalmanPersonalTracker(detector, embedder2, MetricType.MAHALANOBIS_DISTANCE)
    cur = os.getcwd()
    for target in glob.glob(f"{cur}/validates/validate_sets/{args.set}/target_images/*.png"):
        print(target)
        croped_image = cv2.imread(target)
        full_bbox = (0, 0, croped_image.shape[1], croped_image.shape[0])
        tracker1.add_target_features(croped_image, full_bbox)
        tracker2.add_target_features(croped_image, full_bbox)
    print(f"Target added current size {len(tracker1._target_features_pool)}")
    
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
