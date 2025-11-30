import streamlit as st
import torch
import torchaudio
import soundfile as sf
import numpy as np
import os
import datetime
from models.pretrained_asr import PretrainedASRModel
from utils.config import Config

# --- Page Configuration ---
st.set_page_config(
    page_title="Ayamra Hospitals | Intelligent Dictation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Medical Theme ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    h2, h3 {
        color: #34495e;
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Input Fields */
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #ced4da;
    }
    
    /* Primary Button (Record/Transcribe) */
    .stButton button {
        background-color: #008080; /* Teal */
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #006666;
        color: white;
    }
    
    /* Success Message */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    
    /* Metadata Box */
    .metadata-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #008080;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_model():
    """Loads the ASR model from the root directory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "asr_model.pt"
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in root directory!")
        return None, None, device

    try:
        # Initialize the wrapper class
        # Defaulting to whisper-base as per previous context
        model = PretrainedASRModel(model_name='openai/whisper-base')
        
        # Load state dict
        # We use strict=False to be lenient with minor mismatches if any
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle case where state_dict might be nested under 'model_state_dict' or similar
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, device

def save_uploaded_file(uploaded_file):
    try:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# --- Main App Layout ---

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏥 Ayamra Hospitals")
        st.markdown("### Intelligent Medical Dictation System")
    with col2:
        # Auto-updating Date/Time
        now = datetime.datetime.now()
        st.markdown(f"""
            <div style='text-align: right; color: #666;'>
                <b>Date:</b> {now.strftime('%Y-%m-%d')}<br>
                <b>Time:</b> {now.strftime('%H:%M')}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Metadata Section
    st.markdown('<div class="metadata-box">', unsafe_allow_html=True)
    st.subheader("📝 Session Metadata")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        doctor_name = st.text_input("Attending Physician", placeholder="Dr. Name")
        patient_name = st.text_input("Patient Name/ID", placeholder="Patient Name or ID")
    
    with m_col2:
        note_title = st.text_input("Note Title", placeholder="e.g., Initial Consultation")
        context = st.selectbox("Clinical Context", 
                             ["General Consultation", "Diagnosis", "Lab Results", "Prescription", "Follow-up", "Radiology Report", "Surgery Notes"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Loading
    with st.spinner("Initializing AI Engine..."):
        model, device = load_model()

    if model:
        # Recording/Input Section
        st.subheader("🎙️ Dictation")
        
        tab1, tab2 = st.tabs(["Live Recording", "Upload Audio"])
        
        audio_path = None
        
        with tab1:
            # Using st.audio_input if available (Streamlit 1.40+), otherwise fallback or just file uploader
            # Since I can't guarantee the version, I'll use the file uploader as primary fallback 
            # or check if st.audio_input exists.
            # For now, let's assume standard file upload for "recording" if the widget isn't there, 
            # but the user asked for "record". 
            # I will use st.audio_input which is the modern way.
            try:
                audio_value = st.audio_input("Press to Record")
                if audio_value:
                    # Save to temp file
                    with open("temp_recording.wav", "wb") as f:
                        f.write(audio_value.getbuffer())
                    audio_path = "temp_recording.wav"
            except AttributeError:
                st.warning("Your Streamlit version doesn't support native recording. Please upgrade or use Upload.")
        
        with tab2:
            uploaded_file = st.file_uploader("Upload WAV file", type=['wav'])
            if uploaded_file:
                audio_path = save_uploaded_file(uploaded_file)

        # Transcription Logic
        if audio_path:
            st.audio(audio_path)
            
            if st.button("Transcribe Dictation", type="primary"):
                with st.spinner("Transcribing..."):
                    try:
                        # Load audio using soundfile
                        waveform_np, sample_rate = sf.read(audio_path)
                        
                        # Convert to tensor
                        waveform = torch.tensor(waveform_np, dtype=torch.float32)
                        
                        # Handle dimensions (Mono conversion)
                        if waveform.ndim > 1:
                            waveform = waveform.t()
                            waveform = torch.mean(waveform, dim=0)
                        
                        # Resample to 16000Hz
                        if sample_rate != 16000:
                            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                            waveform = resampler(waveform)
                        
                        # Feature Extraction using AutoProcessor
                        if hasattr(model, 'processor'):
                            inputs = model.processor(
                                waveform, 
                                sampling_rate=16000, 
                                return_tensors="pt"
                            ).to(device)
                            
                            # Generate
                            with torch.no_grad():
                                predictions = model.generate(
                                    **inputs,
                                    forced_decoder_ids=None,
                                    task="transcribe"
                                )
                                
                                transcription = model.processor.batch_decode(
                                    predictions,
                                    skip_special_tokens=True
                                )[0]
                                
                                # Display Result
                                st.success("Transcription Complete")
                                st.markdown("### 📄 Transcribed Note")
                                st.markdown(f"""
                                    <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
                                        {transcription}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Download Button
                                note_content = f"""
                                Date: {now.strftime('%Y-%m-%d %H:%M')}
                                Physician: {doctor_name}
                                Patient: {patient_name}
                                Context: {context}
                                Title: {note_title}
                                
                                --- TRANSCRIPT ---
                                {transcription}
                                """
                                st.download_button(
                                    label="Download Note",
                                    data=note_content,
                                    file_name=f"note_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain"
                                )
                                
                        else:
                            st.error("Model processor not found.")
                            
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                        st.write("Debug info:", str(e))

if __name__ == "__main__":
    main()
