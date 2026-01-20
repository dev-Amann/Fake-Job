# 🛡️ AI-Powered Fake Job & Internship Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![App](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Offline%20%26%20Ready-green)

An advanced, **offline Machine Learning application** designed to protect students and job seekers from fraudulent employment offers. This system analyzes job descriptions in real-time to classify them as **Genuine**, **Suspicious**, or **Scam** with **97% accuracy**.

---

## 📌 Problem Statement

In the digital age, online job portals have become the primary source for employment. However, this has led to a significant rise in **employment fraud**, where scammers prey on vulnerable job seekers—especially students and fresh graduates.

Common issues include:
*   **Financial Loss**: Scammers asking for "registration fees" or "equipment deposits".
*   **Identity Theft**: Phishing for sensitive personal information.
*   **Wasted Time**: Students spending effort on fake opportunities.

Existing solutions often require internet connectivity or paid APIs. There is a critical need for a **privacy-focused, local, and explainable AI tool** that can detect these scams instantly without relying on cloud services.

---

## 💡 Solution Overview

This project provides a **end-to-end local AI solution** that utilizes **Natural Language Processing (NLP)** and **Machine Learning** to identify fraudulent patterns in text.

*   **Offline First**: Runs entirely on your local machine; no data leaves your privacy.
*   **Context Aware**: Analyzes the entire job context (Title, Description, Requirements, Company Profile).
*   **Explainable**: Provides confidence scores and highlights specific "red flag" keywords (e.g., "bank transfer", "instant hire").

---

## 🚀 Key Features

*   **⚡ Real-Time Analysis**: Instant classification of job posts.
*   **🎯 High Accuracy (~97%)**: Trained on a massive dataset of 17,880 real-world job records.
*   **📊 Probability Scoring**: Distinguishes between definite Scams and "Suspicious" postings based on confidence levels.
*   **🚩 Risk Indicators**: Automatically flags high-risk keywords like "registration fee," "urgent wire," etc.
*   **💻 Zero Dependencies**: No API keys or internet required for prediction.

---

## 🛠️ Technology Stack

| Component | Technology | Usage |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core logic and scripting |
| **ML Models** | Scikit-Learn | Logistic Regression, TF-IDF Vectorizer |
| **NLP** | NLTK | Text preprocessing, Lemmatization, Stopword removal |
| **Data Processing** | Pandas | Dataset manipulation and cleaning |
| **Frontend** | Streamlit | Interactive web-based user interface |
| **Persistence** | Joblib | Saving and loading trained models |

---

## 📊 Dataset & Model Architecture

The model is trained on the **Employment Scam Aegean Dataset (EMSCAD)**, widely known as `fake_job_postings.csv`.

*   **Total Records**: 17,880
*   **Features Used**: Combined text from `title`, `description`, `requirements`, and `company_profile`.
*   **Algorithm**: Logistic Regression with Class Balancing (to handle the rarity of scam cases).
*   **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) with unigrams and bigrams (`ngram_range=(1,2)`).

---

## ⚙️ How to Run Locally

### 1. Prerequisites
Ensure you have Python installed. Clone the repository and navigate to the folder:

```bash
git clone https://github.com/dev-Amann/Fake-Job.git
cd Fake-Job
```

### 2. Install Dependencies
Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to assist the AI in learning from the dataset. This will generate the `model/` files.

```bash
python train.py
```
*> **Note**: Training may take about 1 minute due to the large dataset size.*

### 4. Launch the Application
Start the Streamlit interface:

```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```bash
job_scam_detector/
├── fake_job_postings.csv      # Large Training Dataset (17.8k records)
├── train.py                   # ML Training Script (Preprocessing + Model Training)
├── app.py                     # Main Streamlit Application
├── requirements.txt           # Project Dependencies
├── README.md                  # Documentation
└── model/                     # Artifacts (Generated after training)
    ├── scam_detector.pkl      # Trained Logistic Regression Model
    └── tfidf_vectorizer.pkl   # Fitted TF-IDF Vectorizer
```

---

## 🛡️ Disclaimer
This tool is intended for educational and awareness purposes. While it achieves high accuracy, users should always exercise their own judgment and due diligence before applying for jobs or making payments.

---
*Built with ❤️ for a safer job search experience.*
