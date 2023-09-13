from datetime import time
import cv2
from cv2.typing import MatLike
import argparse
from playsound import playsound
import threading

from src.detectors import BaseDetector
from src.detectors.segment_detector import SegmentDetector
from src.embedders import BaseEmbedder
from src.embedders.available_embedder_models import AvailableEmbedderModels
from src.embedders.clip_embedder import CLIPEmbedder
from src.metrics.base_metric import MetricType
from src.personal_trackers import TrackResults, PersonalTracker
from src.personal_trackers.me_personal_tracker import MEPersonalTracker
from src.personal_trackers.sift_personal_tracker import SIFTPersonalTracker

class VideoStreamWidget(object):
    def __init__(self, src: str):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.ret = False
        
        

    def read(self):
        # Return the latest frame
        return self.ret, self.frame

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.ret, self.frame) = self.capture.read()


def main(source: str | int, args):
    rtmp_url = "http://192.168.137.232:8080/video"
    cap = cv2.VideoCapture(source)
    # cap = cv2.VideoCapture(rtmp_url)
    # cap = VideoStreamWidget(src=rtmp_url)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 200)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    detector = BaseDetector()
    embedder = BaseEmbedder()
    # tracker = PersonalTracker(detector, embedder, MetricType.CSEM_DISTANCE, auto_add_target_features=args.auto_add_target_features, auto_add_target_features_interval=args.auto_add_target_features_interval)
    tracker = SIFTPersonalTracker(SegmentDetector(), embedder, MetricType.COSINE_SIMILARITY).use_embedder()
    playsound("./cap.mp3")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord(' '):
            break
    targets = tracker.get_repetitive_target_from_camera(cap, 4)
    for target in targets:
        tracker.add_target_features(target[0], target[1])

    cv2.imshow("target", tracker.show_target_images())
    print(f"Start tracking with {len(tracker._target_features_pool)} targets")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            track_result = tracker.track(frame, draw_result=True)
        except:
            continue
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
