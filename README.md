# Ayamra Hospitals Intelligent Dictation System

## Project Overview
This repository contains the source code for an intelligent medical dictation system developed for Ayamra Hospitals. The primary objective of this application is to streamline the clinical documentation process by leveraging advanced Automatic Speech Recognition (ASR) technologies. By integrating a fine-tuned Transformer-based model with a user-friendly web interface, the system allows healthcare professionals to accurately transcribe patient notes, diagnoses, and treatment plans directly from voice input.

## System Architecture and Features
The application is built using a modular architecture that combines a responsive frontend with a powerful deep learning backend.

### User Interface
The frontend is developed using Streamlit, chosen for its ability to rapidly deploy data-centric web applications. We have customized the interface to reflect a professional medical environment, utilizing a clean, card-based layout that prioritizes ease of use. Key interface elements include:
- **Session Metadata**: Dedicated fields for capturing essential context, such as the attending physician's name, patient identification, and the specific clinical context (e.g., General Consultation, Radiology Report).
- **Dual Input Modes**: To accommodate different workflows, the system supports both live recording directly through the browser and the upload of pre-recorded WAV audio files.

### ASR Engine
At the heart of the system lies the ASR engine, which utilizes the OpenAI Whisper architecture (specifically the Tiny variant). This model has been selected for its balance of inference speed and transcription accuracy. The implementation handles the complete audio processing pipeline:
1.  **Audio Preprocessing**: Raw audio input is converted into tensors and resampled to 16,000 Hz to match the model's expected input requirements.
2.  **Inference**: The processed audio is passed through the Transformer model to generate text transcriptions.
3.  **Post-processing**: The raw text is formatted and presented to the user for review.

## Technical Requirements
To run this application successfully, the following prerequisites must be met:
-   **Python Environment**: Python 3.8 or higher is required.
-   **Hardware Acceleration**: While the model can run on a CPU, a CUDA-enabled GPU is highly recommended to ensure real-time transcription performance.
-   **Model Checkpoint**: The system requires a specific model checkpoint file named `asr_model.pt` to be present in the root directory. This file contains the weights for the fine-tuned Whisper model.

## Installation and Setup Instructions
Follow these steps to set up the development environment and launch the application.

1.  **Repository Setup**
    Clone the project repository to your local machine and navigate into the project directory.

2.  **Dependency Management**
    We recommend using a virtual environment to manage project dependencies. Install the required Python packages using the provided requirements file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Configuration**
    Ensure that the trained model file (`asr_model.pt`) is placed directly in the root folder of the project. The application is configured to look for this specific filename during initialization.

4.  **Launching the Application**
    Start the Streamlit server by running the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    Once the server starts, the application will be accessible via your web browser at the local address provided (typically `http://localhost:8501`).

## Usage Guidelines
Upon launching the application, users are presented with a dashboard to enter session details. After populating the metadata, users can proceed to the Dictation section.
-   For **Live Recording**, ensure your microphone permissions are enabled, then record your dictation.
-   For **File Upload**, simply drag and drop a valid WAV file into the upload area.

After capturing the audio, click the "Transcribe Dictation" button. The system will process the audio and display the transcribed text. This text, along with the session metadata, can then be downloaded as a formatted text file for inclusion in the patient's medical record.
