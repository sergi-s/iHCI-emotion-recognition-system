import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image

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

def detect_emotion_from_frame(frame):
    """Process a single frame and return detected emotion"""
    # Convert to grayscale and back to 3-channel for YOLO
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3channel = cv2.merge([gray, gray, gray])
    
    # Run YOLO inference
    results = model(gray_3channel)
    result = results[0]
    
    emotion = "Neutral"
    confidence = 0.0
    annotated_frame = frame
    
    # Process detections
    if result.boxes and len(result.boxes) > 0:
        boxes = result.boxes.data.cpu().numpy()
        if len(boxes) > 0:
            # Get detection with highest confidence
            best_detection = boxes[np.argmax(boxes[:, 4])]
            class_id = int(best_detection[5])
            confidence = best_detection[4]
            
            if confidence > 0.3:  # Lower threshold for demo
                emotion = result.names[class_id]
    
    # Get annotated frame
    try:
        annotated_frame = result.plot()
    except:
        pass
    
    return emotion, confidence, annotated_frame

def main():
    st.set_page_config(
        page_title="Smart Vehicle Emotion Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Smart Vehicle Emotion Detection System")
    st.markdown("---")
    
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        # Camera input
        camera_input = st.camera_input("Take a photo to detect your emotion")
        
        if camera_input is not None:
            # Convert the uploaded image to OpenCV format
            image = Image.open(camera_input)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect emotion
            emotion, confidence, annotated_frame = detect_emotion_from_frame(frame)
            
            # Convert back to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the annotated frame
            st.image(annotated_frame_rgb, caption="Emotion Detection Result", use_container_width=True)
            
            # Store in session state
            st.session_state.current_emotion = emotion
            st.session_state.current_confidence = confidence
        
        # Alternative: File uploader for testing
        st.markdown("### Or upload an image for testing")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            emotion, confidence, annotated_frame = detect_emotion_from_frame(frame)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_frame_rgb, caption="Emotion Detection Result", use_container_width=True)
            
            st.session_state.current_emotion = emotion
            st.session_state.current_confidence = confidence
    
    with col2:
        st.subheader("🎭 Vehicle Dashboard")
        
        # Initialize session state
        if 'current_emotion' not in st.session_state:
            st.session_state.current_emotion = "Neutral"
        if 'current_confidence' not in st.session_state:
            st.session_state.current_confidence = 0.0
        
        # Current emotion display
        current_emoji = EMOTION_EMOJIS.get(st.session_state.current_emotion, "😐")
        
        # Large emoji display
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <div style="font-size: 120px; margin-bottom: 10px;">{current_emoji}</div>
                <h2 style="color: #1f77b4;">Driver: {st.session_state.current_emotion}</h2>
                <p style="font-size: 18px;">Confidence: {st.session_state.current_confidence:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Vehicle status based on emotion
        st.markdown("### 🚗 Vehicle Response")
        
        if st.session_state.current_emotion == "Angry":
            st.error("⚠️ Driver Alert: High stress detected")
            vehicle_action = "🎵 Activating calming environment"
            st.markdown("🌡️ Adjusting climate control")
            st.markdown("🔊 Reducing audio volume")
        elif st.session_state.current_emotion == "Fearful":
            st.warning("😰 Safety Mode: Anxiety detected")
            vehicle_action = "🛡️ Enhanced safety protocols"
            st.markdown("🚨 Increasing alert systems")
            st.markdown("📱 Ready to contact emergency services")
        elif st.session_state.current_emotion == "Happy":
            st.success("😊 Optimal State: Happy driver")
            vehicle_action = "✅ All systems normal"
            st.markdown("🎶 Maintaining current playlist")
            st.markdown("🌟 Cruise control available")
        elif st.session_state.current_emotion == "Sad":
            st.info("😢 Comfort Mode: Low mood detected")
            vehicle_action = "🌈 Mood enhancement active"
            st.markdown("💡 Adjusting cabin lighting")
            st.markdown("🎵 Playing uplifting music")
        else:  # Neutral
            st.info("😐 Standard Mode: Neutral state")
            vehicle_action = "🚗 Normal operation"
            st.markdown("⚙️ Standard settings active")
            st.markdown("📊 Monitoring driver state")
        
        st.markdown(f"**🎯 Current Action:** {vehicle_action}")
        
        # Vehicle metrics
        st.markdown("### 📊 Vehicle Metrics")
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Speed", "65 mph", "0")
            st.metric("Fuel", "75%", "-2%")
        with col2b:
            st.metric("Temperature", "72°F", "2°F")
            st.metric("Safety Score", "95/100", "+5")
        
        # Quick recommendations
        st.markdown("### 💡 Quick Actions")
        recommendations = {
            "Angry": "🎵 Play relaxing music",
            "Fearful": "🛡️ Activate safety mode", 
            "Happy": "🌟 Enable cruise control",
            "Neutral": "📱 Check navigation",
            "Sad": "🌈 Brighten cabin lights"
        }
        
        action = recommendations.get(st.session_state.current_emotion, "🚗 Continue driving")
        if st.button(action, use_container_width=True):
            st.success(f"✅ {action} activated!")
        
        # Emergency button
        st.markdown("---")
        if st.button("🚨 Emergency Stop", use_container_width=True, type="secondary"):
            st.error("🛑 Emergency protocols activated!")

if __name__ == "__main__":
    main()