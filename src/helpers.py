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
def rec_check(recs: list[tuple[int, int, int, int]], index, threshold):
    def cal_area (target, compare):
        x_overlap = 0
        y_overlap = 0

        tx_start = target[0][0]
        tx_end = target[1][0]
        cx_start = compare[0][0]
        cx_end = compare[1][0]

        ty_start = target[0][1]
        ty_end = target[1][1]
        cy_start = compare[0][1]
        cy_end = compare[1][1]

        if cx_start <= tx_end and cx_end >= tx_end:
            x_overlap = tx_end - cx_start
        elif cx_end >= tx_start and cx_start <= tx_start:
            x_overlap = cx_end - tx_start
        elif cx_start >= tx_start and cx_end <= tx_end:
            x_overlap = cx_end - cx_start
        elif cx_start <= tx_start and cx_end >= tx_end:
            x_overlap = tx_end - tx_start

        if cy_start <= ty_end and cy_end >= ty_end:
            y_overlap = ty_end - cy_start
        elif cy_end >= ty_start and cy_start <= ty_start:
            y_overlap = cy_end - ty_start
        elif cy_start >= ty_start and cy_end <= ty_end:
            y_overlap = cy_end - cy_start
        elif cy_start <= ty_start and cy_end >= ty_end:
            y_overlap = ty_end - ty_start

        return x_overlap * y_overlap
    
    recAll = []
    area = 0
    percent_area = 0

    for r in recs:
        recAll.append([(r[0], r[1]), (r[2], r[3])])

    size_target = cal_area(recAll[index], recAll[index])

    for i in range(0, len(recAll)):
        if i == index:
            pass
        else:
            area = cal_area(recAll[index], recAll[i])
            percent_area = area/size_target * 100

        if (percent_area >= threshold):
            return False

    return True
