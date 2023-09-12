from ultralytics.engine.results import Boxes, Results, Keypoints

class DetectorResult:
    def __init__(self, ultralytics_result: Results) -> None:
        ultralytics_result = ultralytics_result.cpu()
        result_boxes = ultralytics_result.boxes
        assert result_boxes is not None
        assert len(result_boxes) > 0
        self.results_boxes = result_boxes
        self.keypoints = ultralytics_result.keypoints
        
    def __len__(self) -> int:
        return len(self.results_boxes)
    def __getitem__(self, index: int) -> tuple[tuple[int, int, int, int], float, int]:
        return self.bboxes[index], self.confidences[index], self.classes[index]

    @property
    def bboxes(self) -> list[tuple[int, int, int, int]]:
        bboxes = self.results_boxes.xyxy.tolist()
        return [tuple(map(int, bbox)) for bbox in bboxes]
    
    @property
    def confidences(self) -> list[float]:
        return self.results_boxes.conf.tolist()
    
    @property
    def classes(self) -> list[int]:
        return self.results_boxes.cls.tolist()


    def is_front(self, idx: int) -> bool:
        assert self.keypoints is not None
        assert len(self.keypoints) > idx
        assert self.keypoints.conf is not None
        confs = self.keypoints.conf
        assert len(confs) > idx
        conf = confs[idx]
        assert len(conf) == 17
        return all([float(conf[i]) >= 0.8 for i in [0, 1, 2]]) and (abs(float(conf[5])-float(conf[6])) <= 0.2)

    def is_back(self, idx: int) -> bool:
        assert self.keypoints is not None
        assert len(self.keypoints) > idx
        return not self.is_front(idx) and not self.is_side(idx)

    def is_side(self, idx: int) -> bool:
        assert self.keypoints is not None
        assert len(self.keypoints) > idx
        assert self.keypoints.conf is not None
        confs = self.keypoints.conf
        assert len(confs) > idx
        conf = confs[idx]
        if conf[0] >= 0.8 and (conf[1] >= 0.8 or conf[2] >= 0.8) and (conf[3] >= 0.8 or conf[4] >= 0.8):
            return True
        else:
            return False
        
        

    def direction(self, idx: int) -> str:
        assert self.keypoints is not None
        assert len(self.keypoints) > idx
        if self.is_front(idx):
            return "FRONT"
        if self.is_side(idx):
            return "SIDE"
        else:
            return "BACK"

    @property
    def head_bboxes(self) -> list[tuple[int, int, int, int]]:
        assert self.keypoints is not None
        raise NotImplementedError()
