# Fight-detection-and-womens-safety
1. Video Input Acquisition
The system receives live video streams from surveillance sources such as CCTV cameras or uploaded video files.

2. Frame Extraction
The video stream is divided into sequences of frames (e.g., 16 consecutive frames) to capture motion and temporal context.

3. Preprocessing
Each frame is resized, normalized, and optionally augmented (brightness, occlusion simulation) to improve model robustness.

4. Spatial Feature Extraction (CNN)
A Convolutional Neural Network (e.g., ResNet-50) processes each frame to extract high-level spatial features representing visual patterns.

5. Temporal Sequence Modeling (LSTM)
The extracted features from the sequence of frames are passed through an LSTM (Long Short-Term Memory) network to learn motion dynamics and identify violent activity over time.

6. Classification
The LSTM outputs a probability indicating the likelihood of a fight occurring in the frame sequence.

7. Decision Thresholding
If the fight probability exceeds a predefined threshold (e.g., 0.35), the system flags it as a violent incident.

8. Real-Time Alerting
Upon detection, the system triggers alerts via configured methods such as email, SMS, MQTT, or dashboard notifications for immediate action.

9. Result Logging & Visualization
Detected events are logged with timestamps and can be visualized on a dashboard or saved as annotated video clips for review.

10. Post-Processing and Adaptation
The model can be fine-tuned for specific environments (e.g., low light, crowded scenes) or extended to detect other violent behaviors in future updates.
