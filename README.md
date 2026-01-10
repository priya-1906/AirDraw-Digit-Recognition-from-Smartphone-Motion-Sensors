**AirDraw – Digit Recognition from Smartphone Motion Sensors
**
**Project Overview
**
AirDraw is a beginner-level deep learning project that recognizes digits (0–9) written in the air using smartphone IMU sensor data (accelerometer and gyroscope).

This project demonstrates the complete machine learning workflow including:
>data preprocessing
>model training
>model evaluation
>simple deployment using Streamlit

**The final system supports:
**
>Offline prediction using CSV files
>Real-time prediction using live IMU sensor data
>Running the application using Python or Docker

**Dataset Summary
**
Source: Custom dataset collected from smartphone IMU sensors
Sensors Used: Accelerometer and Gyroscope (Magnetometer optional)
Classes: 10 digits (0–9)
Data Type: Multivariate time-series data
Sequence Length: Each sample is resampled to 200 time steps
Features:
  6 channels (accelerometer + gyroscope)
Data Collection Details:
  Each user wrote each digit 20 times
  Total users: 3
  Total samples:10 digits × 20 samples × 3 users = 600 samples

**Project Structure
**
IMU-Air-Digit-Recognition/
├── app/            # Streamlit demo application
├── training/       # Training and evaluation code
├── models/         # Saved trained models
├── scalers/        # Saved scaling files
├── data/           # Raw and processed datasets
├── results/        # Confusion matrices and reports
├── docker/         # Docker files (optional)
├── reports/        # Project reports and documents
└── requirements.txt

**Training & Evaluation**

>Stratified train / validation / test split
>Feature scaling fitted only on training data (to avoid data leakage)

**Data augmentation techniques used:
**
>Gaussian noise
>Minor temporal scaling
>Evaluation Metrics
>Accuracy
>Precision (macro)
>Recall (macro)
>F1-score (macro)
>Confusion matrix
All trained models and evaluation results are saved automatically.

**Deployment
**
The project is deployed using Docker to ensure easy and consistent execution across systems.
The application can also be run directly using Python and Streamlit, and a Windows one-click script is provided for convenience.

**Real-Time Demo (AirDraw App)**

The Streamlit application supports:
  Live IMU sensor streaming from a smartphone
  CSV file upload for offline testing
  Model selection (CNN or GRU)
  Real-time visualization of sensor signals
  Instant digit prediction display

**System Workflow**
Smartphone IMU Data
        ↓
Data Buffering
        ↓
Preprocessing & Resampling
        ↓
Feature Scaling
        ↓
Trained Model (CNN / GRU)
        ↓
Digit Prediction (0–9)
        ↓
Streamlit Interface

Conclusion

This project helped me understand how to work with time-series sensor data, train deep learning models, evaluate results, and deploy a simple application.
It serves as a beginner-friendly implementation of an IMU-based air-written digit recognition system.
