import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

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

def train_model():
    print("Loading dataset from 'fake_job_postings.csv'...")
    try:
        df = pd.read_csv('fake_job_postings.csv')
    except FileNotFoundError:
        print("Error: 'fake_job_postings.csv' not found in root directory.")
        return

    print(f"Dataset loaded. Shortlisting features from {len(df)} records...")

    df.fillna('', inplace=True)

    print("Combining text columns...")
    df['text'] = df['title'] + " " + df['description'] + " " + df['requirements'] + " " + df['company_profile']
    


    df['label'] = df['fraudulent'].map({0: 'genuine', 1: 'scam'})
    
    print("Preprocessing text (this might take a while)...")

    df['clean_text'] = df['text'].apply(preprocess_text)
    
    print("Extracting features using TF-IDF...")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    print("Training Logistic Regression model...")

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Saving Model
    if not os.path.exists('model'):
        os.makedirs('model')
        
    print("Saving model and vectorizer...")
    joblib.dump(model, 'model/scam_detector.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
    print("Done!")

if __name__ == "__main__":
    train_model()
