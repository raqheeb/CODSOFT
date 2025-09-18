# spam_detection.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_data(path='data/spam.csv'):
    # try common encodings & column names
    df = pd.read_csv(path, encoding='latin-1')
    cols = df.columns.tolist()
    if 'label' in cols and 'message' in cols:
        df = df[['label','message']]
    elif 'v1' in cols and 'v2' in cols:
        df = df[['v1','v2']].rename(columns={'v1':'label','v2':'message'})
    else:
        # fallback: take first two columns
        df = df.iloc[:, :2]
        df.columns = ['label','message']
    df = df.dropna()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['message'] = df['message'].astype(str)
    return df

def train_and_save(df, model_path='models/spam_pipeline.joblib'):
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=2),
        MultinomialNB()
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred, labels=['ham','spam']))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to: {model_path}")
    return pipeline

if __name__ == '__main__':
    df = load_data('data/spam.csv')
    model = train_and_save(df)
    # quick inference example
    sample = ["Win a brand new phone! Reply YES to claim", "Hey, are you coming at 7?"]
    print("\nSample predictions:")
    for s,p in zip(sample, model.predict(sample)):
        print(f"{p}  -  {s}")
