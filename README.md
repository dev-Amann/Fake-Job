# AI Fake Job & Internship Scam Detector

## Project Overview
This project is an **Offline Machine Learning Application** designed to classify job and internship messages into three categories: **Scam**, **Suspicious**, and **Genuine**. The system helps students and job seekers identify fraudulent offers without relying on cloud APIs or external services.

## Problem Statement
Job scams are rising, targeting students with offers of "high pay for little work" or asking for registration fees. This tool uses NLP and ML to detect such fraudulent patterns locally.

## ML Approach
The project uses **Logistic Regression** with **TF-IDF (Term Frequency-Inverse Document Frequency)** for text classification.

### Why this approach?
- **TF-IDF**: Effectively converts text into numerical features by highlighting important words while ignoring common ones (stopwords).
- **Logistic Regression**: A simple, interpretable, and efficient algorithm for multi-class classification on small-to-medium datasets. It works well for text data where features are high-dimensional.
- **Offline & Lightweight**: Unlike transformers (BERT, GPT), this approach runs instantly on any standard laptop without a GPU or internet connection.

### Tech Stack
- **Python 3.10+**
- **Pandas**: Data handling
- **NLTK**: Text preprocessing (Lemmatization, Stopwords)
- **Scikit-learn**: Model training (Logistic Regression, TF-IDF)
- **Streamlit**: User Interface
- **Joblib**: Model persistence

**NOTE: No cloud services or paid APIs are used in this project.**

## How to Run Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   Run the training script. This script now uses the large `fake_job_postings.csv` dataset (17.8k records).
   ```bash
   python train.py
   ```
   *Note: Training may take a minute due to the larger dataset.*

3. **Run the Application**
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Use the App**
   - Paste a job message in the text box.
   - Click "Analyze".
   - View the prediction and red flags.

## Project Structure
```
job_scam_detector/
├── fake_job_postings.csv      # Dataset (17,880 records)

├── model/                     # Saved Models
│   ├── scam_detector.pkl
│   └── tfidf_vectorizer.pkl
├── train.py                   # Training Script
├── app.py                     # Streamlit Application
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```
