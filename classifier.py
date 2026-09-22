import codecs
import email
import os
import pickle
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
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
    
    # Stratified 80/20 train/test split to evaluate with a Confusion Matrix
    X_train, X_test, y_train, y_test = train_test_split(
        temails['body'],
        temails['category'],
        test_size=0.20,
        random_state=42,
        stratify=temails['category']
    )
    print(f"Split data into {len(X_train)} training samples and {len(X_test)} test samples.")

    print("Fitting CountVectorizer and TF-IDF on training set...")
    count_vect = CountVectorizer(stop_words='english').fit(X_train)
    X_train_counts = count_vect.transform(X_train)

    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    print("Fitting MultinomialNB Classifier...")
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # Evaluate on unseen test data
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("         MODEL EVALUATION ON TEST SET")
    print("=" * 55)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix (labels=['ham', 'spam']):")
    print("                     PREDICTED")
    print("                  Ham         Spam")
    print(f"ACTUAL Ham   |   {tn:<8}    {fp:<8} (TN={tn}, FP={fp})")
    print(f"ACTUAL Spam  |   {fn:<8}    {tp:<8} (FN={fn}, TP={tp})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=['ham', 'spam'], digits=4))
    print("=" * 55 + "\n")

    try:
        from evaluate import show_confusion_matrix_graph
        from sklearn.metrics import precision_score, recall_score, f1_score
        prec_spam = precision_score(y_test, y_pred, pos_label='spam')
        rec_spam = recall_score(y_test, y_pred, pos_label='spam')
        f1_spam = f1_score(y_test, y_pred, pos_label='spam')
        show_confusion_matrix_graph(cm, y_test, acc, prec_spam, rec_spam, f1_spam)
    except Exception as e:
        print(f"Notice: Could not display graph window: {e}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((count_vect, tfidf_transformer, clf), f)
    print("Saved model cache for faster future runs.")

    return count_vect, tfidf_transformer, clf


if __name__ == '__main__':
    import sys
    # If user ran with --matrix or --eval flag, launch confusion matrix graph immediately
    if any(arg in sys.argv for arg in ('--matrix', '-m', '--eval', '-e', '--confusion')):
        from evaluate import evaluate_model
        evaluate_model(show_graph=True)
        sys.exit(0)

    count_vect, tfidf_transformer, clf = train_or_load_model()

    print("\n" + "=" * 55)
    print("Spam-Ham Email Classifier Ready!")
    print("• Type or paste an email body to classify.")
    print("• Type 'matrix' or 'eval' to open the Confusion Matrix graph.")
    print("• Type 'exit' or 'quit' to leave.")
    print("=" * 55 + "\n")

    while True:
        try:
            text = input("Enter Email body: ").strip().lower()
            if text in ('exit', 'quit'):
                print("Exiting...")
                break
            if not text:
                continue
            if text in ('matrix', 'confusion', 'eval', 'metrics', 'graph'):
                from evaluate import evaluate_model
                evaluate_model(show_graph=True)
                continue

            X_predict = count_vect.transform(np.array([text]))
            X_predict_tfidf = tfidf_transformer.transform(X_predict)
            prediction = clf.predict(X_predict_tfidf)
            print(f"-> This looks like a: {prediction[0].upper()}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
