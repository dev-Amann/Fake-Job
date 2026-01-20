import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/scam_detector.pkl')
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please run train.py first.")
        return None, None

model, vectorizer = load_model()


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

RED_FLAGS = ["fee", "payment", "bank", "urgent", "credit card", "money", "transfer", "whatsapp", "telegram"]

def check_red_flags(text):
    found_flags = [word for word in RED_FLAGS if word in text.lower()]
    return found_flags


st.set_page_config(page_title="AI Scam Detector", page_icon="🚫")

st.title("🚫 AI - Fake Job & Internship Scam Detector")
st.markdown("Enter a job or internship message below to check if it's **Genuine**, **Suspicious**, or a **Scam**.")

user_input = st.text_area("Paste Job Message Here:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    elif model is not None and vectorizer is not None:

        processed_text = preprocess_text(user_input)
        
        text_vectorized = vectorizer.transform([processed_text])
        
        probabilities = model.predict_proba(text_vectorized)[0]
        
        scam_prob = 0.0
        genuine_prob = 0.0
        
        scam_index = list(model.classes_).index('scam')
        scam_prob = probabilities[scam_index]
        genuine_prob = 1 - scam_prob

        if scam_prob > 0.8:
            prediction = "scam"
            confidence = scam_prob * 100
        elif scam_prob > 0.3:
            prediction = "suspicious"
            confidence = scam_prob * 100
        else:
            prediction = "genuine"
            confidence = genuine_prob * 100

        st.divider()
        st.subheader("Analysis Result")
        
        if prediction == "scam":
            st.error(f"🚨 Prediction: **SCAM**")
            st.write(f"Confidence (Scam Probability): **{confidence:.2f}%**")
        elif prediction == "suspicious":
            st.warning(f"⚠️ Prediction: **SUSPICIOUS**")
            st.write(f"Confidence (Scam Probability): **{confidence:.2f}%** (Medium Risk)")
            st.info("This message has characteristics of a scam but isn't definitive. Proceed with caution.")
        else:
            st.success(f"✅ Prediction: **GENUINE**")
            st.write(f"Confidence (Genuine Probability): **{confidence:.2f}%**")
            

        flags = check_red_flags(user_input)
        if flags:
            st.write("#### 🚩 Red Flags Detected:")
            st.write(", ".join([f"`{flag}`" for flag in flags]))
        else:
            if prediction == "genuine":
                st.write("No obvious red keywords found.")
