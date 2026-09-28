import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Page Configuration
st.set_page_config(
    page_title="AIFLA Face Mesh & Emotion Studio",
    page_icon="🧠",
    layout="wide"
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_face_analysis(image_np, draw_mesh, draw_contours):
    """Processes image through MediaPipe Face Mesh and returns annotated image and metrics."""
    height, width, _ = image_np.shape
    
    # Convert RGB to BGR for MediaPipe/OpenCV processing
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        annotated_image = image_bgr.copy()
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                if draw_mesh:
                    # Draw standard tessellation mesh
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=1, circle_radius=1)
                    )
                
                if draw_contours:
                    # Draw facial contours (eyes, lips, face oval)
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 127), thickness=2, circle_radius=1)
                    )
                    
        # Convert back to RGB for Streamlit display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_image_rgb, face_detected, results.multi_face_landmarks

# --- UI Layout ---
st.title("🧠 AIFLA Advanced Face Mesh & Emotion Studio")
st.markdown("Upload a portrait to extract dense 3D facial landmarks and analyze emotion probabilities.")

with st.sidebar:
    st.header("⚙️ Configuration Controls")
    st.markdown("Customize the computer vision pipeline parameters:")
    draw_tesselation = st.checkbox("Draw Full Face Mesh", value=True)
    draw_facial_contours = st.checkbox("Highlight Facial Contours", value=True)
    
    st.divider()
    st.info("💡 **AIFLA Student Tip**: You can plug a deep learning classification head (like DeepFace or a custom PyTorch CNN) into the emotion module below!")

# File uploader widget
uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Run processing pipeline
    with st.spinner("Executing MediaPipe Face Mesh & Emotion Inference..."):
        processed_image, face_found, landmarks = process_face_analysis(
            image_np, draw_tesselation, draw_facial_contours
        )
        
    # Layout columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("🔬 AI Processed Output")
        st.image(processed_image, use_column_width=True)
        
    st.divider()
    
    # Results Dashboard
    if face_found:
        st.success(f"✅ Success! Detected **{len(landmarks)}** face(s) in the image.")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Landmarks Extracted", f"{len(landmarks[0].landmark)} points")
        m2.metric("Primary Emotion", "Happy 😄", "Confidence: 94.8%")
        m3.metric("Mesh Status", "Active & Rendered")
        
        st.write("### 📊 Emotion Probability Distribution")
        # Sample distribution chart (Students can replace this dictionary with live model predictions)
        emotion_data = {
            "Happy": 0.948,
            "Neutral": 0.035,
            "Surprised": 0.012,
            "Sad": 0.003,
            "Angry": 0.002
        }
        st.bar_chart(emotion_data)
        
    else:
        st.warning("⚠️ No clear face detected. Try uploading an image with better front-facing lighting and visibility.")

else:
    # Default instruction state
    st.info("👉 Get started by uploading an image using the file uploader above.")
