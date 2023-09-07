import cv2
from cv2.typing import MatLike
import argparse

from src.detectors import BaseDetector
from src.embedders import BaseEmbedder
from src.personal_trackers import BasePersonalTracker


def main(source: str | int):
    cap = cv2.VideoCapture(source)

    detector = BaseDetector()
    embedder = BaseEmbedder()
    tracker = BasePersonalTracker(detector, embedder)

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
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=0, help="source of video path or camera index")
    args = parser.parse_args()
    if args.source.isdigit():
        vid_source = int(args.source)
    else:
        vid_source = args.source
    main(source=vid_source)
    
