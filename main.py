from datetime import time
import cv2
from cv2.typing import MatLike
import argparse
from playsound import playsound
import threading
from personal_tracker import PersonalTracker, TrackerConfig
from personal_tracker.embedder.available_embedder_models import AvailableEmbedderModels
from personal_tracker.helpers import draw_bbox

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

    config = TrackerConfig().set_auto_add_target_features(True, 10).set_sift_history_size(2).set_embedder_model(AvailableEmbedderModels.OSNET_AIN_X1_0)
    tracker = PersonalTracker(config)
    playsound("./cap.mp3")
    targets = tracker.get_repetitive_target_from_camera(cap, 4)
    for target in targets:
        tracker._tracker.add_target_features(target[0], target[1])

    # cv2.imshow("target", tracker._tracker.show_target_images())
    print(f"Start tracking with {len(tracker._tracker._target_features_pool)} targets")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = tracker.track(frame, True)
        if result.success and result.sorted_scores is not None:
            assert result.target_idx is not None
        cv2.imshow('frame', frame)
        cv2.imshow('result', tracker._tracker.show_target_images())
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
