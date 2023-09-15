from datetime import time
import cv2
from cv2.typing import MatLike
import argparse
from personal_tracker.helpers import crop_image_from_bbox
from playsound import playsound
import threading
from personal_tracker import PersonalTracker
from personal_tracker.helpers import draw_bbox
from src.extentions.hand_triger import HandTriggerResult
from personal_tracker.detector import Detector
import math

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


    tracker = PersonalTracker()
    detector = Detector()
    hand_trigger = HandTriggerResult()
    # playsound("./cap.mp3")
    #targets = tracker.get_target_from_camera(cap, 4)
    #for target in targets:
    #    tracker._tracker.add_target_features(target[0], target[1])

    # cv2.imshow("target", tracker._tracker.show_target_images())
    #print(f"Start tracking with {len(tracker._tracker._target_features_pool)} targets")
    target_person = None
    min_diff = 10000
    while True:
        ret, frame = cap.read()
        #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if not ret:
            break

        
        detector_result = detector.detect(frame)
        if detector_result is None:
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
            diff = math.sqrt((target_person[0] - center_person[0])**2 + (target_person[1] - center_person[1])**2)
            diff_text = f"Diff: {diff:.2f}"
            cv2.putText(frame, diff_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            if diff < min_diff:
                min_diff = diff
                target_bbox = bbox
            target_img = crop_image_from_bbox(frame, target_bbox)
            cv2.imshow("target", target_img)
        if target_person and hand_trigger.is_detect:
            pass
            #tracker._tracker.add_target_features(target_img, target_bbox)
            #result = tracker.track(frame, True)

            
                
        
        hand_trigger.show_status(frame)        
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
