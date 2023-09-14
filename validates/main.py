import cv2
import pafy
from cv2.typing import MatLike
import argparse
import os

from personal_tracker.personal_tracker import PersonalTracker


def main(source: str | int, args):
    print(f"Source: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker1 = PersonalTracker()
    tracker2 = PersonalTracker()
    targets = tracker1.get_repetitive_target_from_camera(cap, 5)
    for target in targets:
        tracker1.add_target_features(target[0], target[1])
        tracker2.add_target_features(target[0], target[1])
    
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
