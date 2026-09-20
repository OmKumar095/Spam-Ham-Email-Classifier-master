import codecs
import email
import os
import pickle
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, 'DataSets', 'index', 'train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'classifier_model.pkl')


def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved model from cache...")
        with open(MODEL_PATH, 'rb') as f:
            count_vect, tfidf_transformer, clf = pickle.load(f)
        return count_vect, tfidf_transformer, clf

    print(f"Reading training index from {INDEX_PATH}...")
    emails = pd.read_csv(INDEX_PATH, sep=" ", header=None, names=['category', 'paths'], nrows=5000)
    
    bodies = []
    print("Reading and parsing email bodies...")
    for index in range(len(emails.paths)):
        rel_path = emails.paths[index].lstrip('.').lstrip('/\\')
        filepath = os.path.join(BASE_DIR, "DataSets", rel_path)
        
        cleaned = ""
        try:
            with codecs.open(filepath, "r", encoding='utf-8', errors='ignore') as fp:
                email_text = email.message_from_file(fp)
                texts = ""
                if email_text.is_multipart():
                    for part in email_text.get_payload():
                        if part.get_content_maintype() == 'text':
                            payload = part.get_payload()
                            if isinstance(payload, str):
                                texts += payload + " "
                else:
                    payload = email_text.get_payload()
                    if isinstance(payload, str):
                        texts += payload + " "

                subject = email_text.get('subject', '')
                if subject:
                    texts += " " + subject

                texts = re.sub(r'\n', ' ', texts)
                texts = re.sub(r'_', '', texts)
                texts = re.sub(r'-', ' ', texts)
                texts = re.sub(r'/', ' ', texts)
                texts = re.sub(r':', ' ', texts)
                texts = re.sub(r'\$', '', texts)
                texts = re.sub(r'=', '', texts)
                texts = re.sub(r'<.*?>', '', texts)
                soup = BeautifulSoup(texts.lower(), 'html.parser')
                cleaned = " ".join(soup.stripped_strings)
        except Exception:
            pass

        bodies.append(cleaned)
        if (index + 1) % 1000 == 0:
            print(f"Parsed {index + 1}/{len(emails.paths)} emails...")

    emails['body'] = bodies
    temails = emails[emails['body'] != '']

    print(f"Dataset ready with {len(temails)} samples ({sum(temails['category'] == 'spam')} spam, {sum(temails['category'] == 'ham')} ham).")
    print("Fitting CountVectorizer and TF-IDF...")
    count_vect = CountVectorizer(stop_words='english').fit(temails['body'])
    X_train_counts = count_vect.transform(temails['body'])

    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    print("Fitting MultinomialNB Classifier...")
    clf = MultinomialNB().fit(X_train_tfidf, temails['category'])
    accuracy = clf.score(X_train_tfidf, temails['category'])
    print(f"Classifier trained successfully with Accuracy: {accuracy * 100:.2f}%")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((count_vect, tfidf_transformer, clf), f)
    print("Saved model cache for faster future runs.")

    return count_vect, tfidf_transformer, clf


if __name__ == '__main__':
    count_vect, tfidf_transformer, clf = train_or_load_model()

    print("\n" + "=" * 50)
    print("Spam-Ham Email Classifier Ready!")
    print("Type or paste an email body to classify.")
    print("Type 'exit' or 'quit' to leave.")
    print("=" * 50 + "\n")

    while True:
        try:
            text = input("Enter Email body: ").strip().lower()
            if text in ('exit', 'quit'):
                print("Exiting...")
                break
            if not text:
                continue

            X_predict = count_vect.transform(np.array([text]))
            X_predict_tfidf = tfidf_transformer.transform(X_predict)
            prediction = clf.predict(X_predict_tfidf)
            print(f"-> This looks like a: {prediction[0].upper()}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
