import streamlit as st
import cv2
from personal_tracker import PersonalTracker
from personal_tracker.embedder.available_embedder_models import AvailableEmbedderModels
from personal_tracker.metric.metric_type import MetricType
from personal_tracker.tracker.tracker_config import TrackerConfig
from personal_tracker.helpers import draw_bbox, draw_center_vertical_line
from streamlit_server_state import server_state, server_state_lock

st.set_page_config(
    page_title="Boombot - WebUI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state='collapsed',
)

# Setup states

@st.cache_resource()
def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

@st.cache_resource()
def init_tracker(_config: TrackerConfig):
    config = TrackerConfig().set_embedder_model(AvailableEmbedderModels.OSNET_AIN_X1_0).set_metric_type(MetricType.COSINE_SIMILARITY2) \
            .set_sift_history_size(2)
    return PersonalTracker(config)

def toggle_tracking():
    st.session_state.start_tracking = not st.session_state.start_tracking
    if not st.session_state.start_tracking:
        del st.session_state.targets
        del st.session_state.has_target
        cam = init_camera()
        cam.release()
        st.cache_resource.clear()

if "start_tracking" not in st.session_state:
    st.session_state.start_tracking = False
    
if "has_target" not in st.session_state:
    st.session_state.has_target = False
# Tracker Config
sidebar = st.sidebar
with sidebar:
    st.title("Boombot")
    st.write("A personal tracker for you")
    st.write("Made with ❤️ by [Boombot Team]")

    embedder_model = st.selectbox("Embedder Model", map(lambda x: x.value,list(AvailableEmbedderModels)))
    tracker_metric = st.selectbox("Tracker Metric", map(lambda x: x.value,list(MetricType)))
    sift_history_size = st.slider("SIFT History Size", 0, 10, 2)
    auto_add_target_features = st.checkbox("Auto Add Target Features", value=False)
    if auto_add_target_features:
        auto_add_target_features_interval = st.slider("Auto Add Target Features Interval in second.", 1, 1000, 1000)

    t_config = TrackerConfig().set_embedder_model(AvailableEmbedderModels(embedder_model)).set_sift_history_size(sift_history_size) \
    .set_metric_type(MetricType(tracker_metric))
    if auto_add_target_features:
        t_config.set_auto_add_target_features(True, auto_add_target_features_interval) # type: ignore


if st.session_state.start_tracking and not st.session_state.has_target:
    if "targets" not in st.session_state:
        st.session_state.targets = []
    st.write("Please select a target")
    cap = init_camera()
    tracker = init_tracker(t_config)
    c1, c2 = st.columns(2)
    with c1:
        add = st.button("Add Target")
        cam_placeholder = st.empty()
    with c2:
        cancel = st.button("Cancel", on_click=toggle_tracking)
        more_than_zero = len(st.session_state.targets) > 0
        start_t = st.button("Start Tracking!", disabled= not more_than_zero, 
                            help="You need at least 1 targets to start tracking" if not more_than_zero else "Click to start tracking")
        if more_than_zero > 0:
            n = 5
            groups = []
            for i in range(0, len(st.session_state.targets), n):
                groups.append(st.session_state.targets[i:i+n])

            cols = st.columns(n)
            for group in groups:
                for i, target in enumerate(group):
                        croped = target[0][target[1][1]:target[1][3], target[1][0]:target[1][2]]
                        resized = cv2.resize(croped, (128, 256))
                        cols[i].image(resized, channels="BGR")

    while True:
        if cancel:
            break
        ret, frame = cap.read()
        if not ret:
            st.write("Unable to read frame")
            break
        result = tracker._detector.detect(frame)
        if result is None:
            continue
        target_bbox = result.bboxes[0]
        croped = frame[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] # y1:y2, x1:x2
        resized = cv2.resize(croped, (128, 256))
        cam_placeholder.image(resized, channels="BGR")
        if start_t:
            for target in st.session_state.targets:
                tracker.add_target_features(*target)
                
            st.session_state.has_target = True
            break
        if add:
            st.session_state.targets.append((frame, target_bbox))
            break
    st.experimental_rerun()
        
if st.session_state.start_tracking:
    if st.button("Stop Tracking", on_click=toggle_tracking):
        st.experimental_rerun()
else:
    if st.button("Start Tracking", on_click=toggle_tracking):
        pass
    st.write("Tracking is off")

    
if st.session_state.start_tracking:
    cap = init_camera()
    cam_placeholder = st.empty()
    tracker = init_tracker(t_config)
    st.write(f"Tracker: started with {len(tracker._tracker._target_features_pool)} targets")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Unable to read frame")
            break
        
        result = tracker.track(frame)
        if not result.success:
            cam_placeholder.image(frame, channels="BGR")
            continue

        assert result.target_bbox is not None
        assert result.sorted_scores is not None
        draw_bbox(frame, result.target_bbox, (0, 255, 0), str(result.sorted_scores[0]))
        frame = draw_center_vertical_line(frame, (255, 255, 255))
        cam_placeholder.image(frame, channels="BGR")
