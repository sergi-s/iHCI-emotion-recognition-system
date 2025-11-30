import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from ultralytics import YOLO
import av
import threading
import time

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('best.onnx')

model = load_model()

# Emotion to emoji mapping
EMOTION_EMOJIS = {
    'Angry': '😡',
    'Fearful': '😨', 
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢'
}

# Shared state for emotion data
class EmotionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_emotion = "Neutral"
        self.confidence = 0.0
        self.last_update = time.time()
    
    def update(self, emotion, confidence):
        with self.lock:
            self.current_emotion = emotion
            self.confidence = confidence
            self.last_update = time.time()
    
    def get(self):
        with self.lock:
            return self.current_emotion, self.confidence, self.last_update

# Global emotion state
emotion_state = EmotionState()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "Neutral"
        self.confidence = 0.0
        self.frame_interval = 1  # Process every 15th frame (slower sampling)
        self.frame_counter = 0
        self.last_processed_time = time.time()
    
    def recv(self, frame):
        self.frame_counter += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Only process every Nth frame for emotion detection
        if self.frame_counter % self.frame_interval == 0:
            # Convert to grayscale and back to 3-channel for YOLO
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_3channel = cv2.merge([gray, gray, gray])
            
            # Run YOLO inference
            results = model(gray_3channel)
            result = results[0]
            
            # Process detections - only update if something is detected
            if result.boxes and len(result.boxes) > 0:
                # Get the detection with highest confidence
                boxes = result.boxes.data.cpu().numpy()
                if len(boxes) > 0:
                    # boxes format: [x1, y1, x2, y2, confidence, class_id]
                    best_detection = boxes[np.argmax(boxes[:, 4])]  # Highest confidence
                    class_id = int(best_detection[5])
                    confidence = best_detection[4]
                    
                    if confidence > 0.1:  # Lower threshold for better responsiveness
                        detected_emotion = result.names[class_id]
                        self.current_emotion = detected_emotion
                        self.confidence = confidence
                        
                        print("=============")
                        print("Detected Emotion:", detected_emotion)
                        print("Confidence:", confidence)
                        print("=============")
                        
                        # Update global state only when detected
                        emotion_state.update(detected_emotion, confidence)
            # If no detection, keep the previous emotion (don't reset)
            
            # Draw bounding boxes and labels on the image
            try:
                annotated_img = result.plot()
                return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
            except:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            # Return frame without processing
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
def main():
    st.set_page_config(
        page_title="Smart Vehicle Emotion Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Smart Vehicle Emotion Detection System")
    st.markdown("---")
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    # WebRTC Configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Initialize session state
    if 'emotion_detected' not in st.session_state:
        st.session_state.emotion_detected = "Neutral"
    if 'confidence_level' not in st.session_state:
        st.session_state.confidence_level = 0.0
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = 0
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )
    
    with col2:
        st.subheader("🎭 Vehicle Dashboard")
        
        # Create placeholder for dynamic updates
        emotion_display = st.empty()
        status_section = st.empty()
        stats_section = st.empty()
        recommendations_section = st.empty()
    
    # Continuous update loop
    while True:
        # Get the latest emotion from global state
        current_emotion, current_confidence, last_update = emotion_state.get()
        
        # Update session state if there's new data
        if last_update > st.session_state.last_update_time:
            st.session_state.emotion_detected = current_emotion
            st.session_state.confidence_level = current_confidence
            st.session_state.last_update_time = last_update
            print(f"UI Updated - Emotion: {current_emotion}, Confidence: {current_confidence:.2f}")
        
        # Current emotion display
        current_emoji = EMOTION_EMOJIS.get(st.session_state.emotion_detected, "😐")
        
        # Large emoji display
        emotion_display.markdown(
            f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 100px;">{current_emoji}</div>
                <h2>Driver Emotion: {st.session_state.emotion_detected}</h2>
                <p>Confidence: {st.session_state.confidence_level:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Vehicle status based on emotion
        with status_section.container():
            st.markdown("### 🚗 Vehicle Status")
            
            if st.session_state.emotion_detected == "Angry":
                st.error("⚠️ Driver appears angry - Consider a break or calming music")
                vehicle_action = "🎵 Playing calming music"
            elif st.session_state.emotion_detected == "Fearful":
                st.warning("😰 Driver appears anxious - Activating safety mode")
                vehicle_action = "🛡️ Enhanced safety alerts active"
            elif st.session_state.emotion_detected == "Happy":
                st.success("😊 Driver is happy - All systems normal")
                vehicle_action = "✅ Optimal driving conditions"
            elif st.session_state.emotion_detected == "Sad":
                st.info("😢 Driver appears sad - Mood enhancement mode")
                vehicle_action = "🌟 Playing upbeat music"
            else:  # Neutral
                st.info("😐 Driver appears neutral - Standard mode")
                vehicle_action = "🚗 Normal operation"
            
            st.markdown(f"**Current Action:** {vehicle_action}")
        
        # Simple stats
        with stats_section.container():
            st.markdown("### 📊 Session Stats")
            st.metric("Current Emotion", st.session_state.emotion_detected)
            st.metric("Detection Confidence", f"{st.session_state.confidence_level:.1%}")
        
        # Vehicle recommendations
        with recommendations_section.container():
            st.markdown("### 💡 Recommendations")
            recommendations = {
                "Angry": ["Take a 5-minute break", "Practice deep breathing", "Lower music volume"],
                "Fearful": ["Reduce speed", "Increase following distance", "Consider pulling over"],
                "Happy": ["Maintain current state", "Stay alert", "Enjoy the drive!"],
                "Neutral": ["Stay focused", "Monitor alertness", "Take breaks as needed"],
                "Sad": ["Listen to uplifting music", "Consider calling a friend", "Take a short break"]
            }
            
            current_recommendations = recommendations.get(st.session_state.emotion_detected, ["Drive safely"])
            for rec in current_recommendations:
                st.markdown(f"• {rec}")
        
        # Update every second
        time.sleep(1.0)
        st.rerun()

if __name__ == "__main__":
    main()