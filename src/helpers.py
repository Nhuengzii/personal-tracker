import cv2
from cv2.typing import MatLike

def crop_image_from_bbox(cv_image: MatLike, bbox: tuple[int, int, int, int]) -> MatLike:
    croped_img = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return croped_img

def draw_bbox(cv_image: MatLike, bbox: tuple[int, int, int, int], color: tuple[int, int, int] = (0, 0, 255), text: str | None = None ):
    cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    if text is not None:
        cv2.putText(cv_image, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def draw_center_vertical_line(cv_image: MatLike, color: tuple[int, int, int]):
    height, width, _ = cv_image.shape
    cv_image = cv2.line(cv_image, (width//2, 0), (width//2, height), color, 2)
    return cv_image
    
def get_target_from_camera(cap) -> tuple[MatLike, tuple[int, int, int, int]]:
    target_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). continue ...")
            continue
        # write tutorial pressed s to capture target
        cv2.putText(frame, "Press s to capture target", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("target", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            target_frame = frame
            break
    # Write tutorial
    
    soi = cv2.selectROI("target", target_frame) 
    # convert to x1, y1, x2, y2
    soi = (int(soi[0]), int(soi[1]), int(soi[0]) + int(soi[2]), int(soi[1]) + int(soi[3]))
    cv2.destroyAllWindows()
    return target_frame, soi
    
