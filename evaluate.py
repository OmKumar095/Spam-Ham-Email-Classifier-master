import codecs
import email
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, 'DataSets', 'index', 'train.csv')
CACHE_PATH = os.path.join(BASE_DIR, 'DataSets', 'clean_emails_cache.pkl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Screenshots')


def load_and_clean_data(index_path, use_cache=True):
    if use_cache and os.path.exists(CACHE_PATH):
        print(f"Loading pre-parsed dataset from cache: {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            clean_emails = pickle.load(f)
        print(f"Loaded {len(clean_emails)} emails from cache successfully.")
        return clean_emails

    print(f"Loading dataset index from {index_path}...")
    emails = pd.read_csv(index_path, sep=" ", header=None, names=['category', 'paths'])
    print(f"Found {len(emails)} entries in index.")

    bodies = []
    print("Reading and parsing email contents...")
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
        if (index + 1) % 1000 == 0 or (index + 1) == len(emails.paths):
            print(f"Parsed {index + 1}/{len(emails.paths)} emails...")

    emails['body'] = bodies
    clean_emails = emails[emails['body'] != ''][['category', 'body']]

    # Save cache for instant future loads
    try:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(clean_emails, f)
        print(f"Cached parsed dataset to {CACHE_PATH} for instant loading.")
    except Exception as e:
        print(f"Cache save notice: {e}")

    return clean_emails


def show_confusion_matrix_graph(cm, y_test, acc, prec, rec, f1):
    tn, fp, fn, tp = cm.ravel()
    total_ham = tn + fp
    total_spam = fn + tp

    fig, ax = plt.subplots(figsize=(7.5, 6), num="Spam-Ham Classifier - Confusion Matrix")

    annot_labels = np.array([
        [f"True Negative (TN)\n{tn}\n({tn/total_ham*100:.1f}%)", f"False Positive (FP)\n{fp}\n({fp/total_ham*100:.1f}%)"],
        [f"False Negative (FN)\n{fn}\n({fn/total_spam*100:.1f}% Missed!)", f"True Positive (TP)\n{tp}\n({tp/total_spam*100:.1f}% Caught)"]
    ])

    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt='',
        cmap='Blues',
        xticklabels=['Predicted Ham', 'Predicted Spam'],
        yticklabels=['Actual Ham', 'Actual Spam'],
        cbar=False,
        annot_kws={'size': 12, 'weight': 'bold'},
        ax=ax,
        linewidths=2,
        linecolor='white'
    )
    ax.set_title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=15, weight='bold', pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=12, weight='bold', labelpad=10)

    plt.tight_layout()

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved confusion matrix image to: {plot_path}")

    # Launch GUI window
    print("Displaying Confusion Matrix Graph window... (Close the window when done)")
    plt.show()


def evaluate_model(show_graph=True):
    emails = load_and_clean_data(INDEX_PATH, use_cache=True)
    total_samples = len(emails)
    ham_count = sum(emails['category'] == 'ham')
    spam_count = sum(emails['category'] == 'spam')

    print("\n" + "=" * 65)
    print("                DATASET DISTRIBUTION")
    print("=" * 65)
    print(f"Total Valid Emails : {total_samples}")
    print(f"Legitimate (Ham)  : {ham_count} ({ham_count / total_samples * 100:.1f}%)")
    print(f"Spam Emails        : {spam_count} ({spam_count / total_samples * 100:.1f}%)")

    # 80/20 Stratified train/test split to preserve distribution
    X_train, X_test, y_train, y_test = train_test_split(
        emails['body'],
        emails['category'],
        test_size=0.20,
        random_state=42,
        stratify=emails['category']
    )

    print(f"\nTraining set size : {len(X_train)} emails")
    print(f"Testing set size  : {len(X_test)} emails (unseen evaluation data)")

    # Feature extraction
    print("\nExtracting features using CountVectorizer & TF-IDF...")
    count_vect = CountVectorizer(stop_words='english')
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train model
    print("Training MultinomialNB classifier...")
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # Evaluate on unseen test data
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)

    # Compute Confusion Matrix
    labels = ['ham', 'spam']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    prec_spam = precision_score(y_test, y_pred, pos_label='spam')
    rec_spam = recall_score(y_test, y_pred, pos_label='spam')
    f1_spam = f1_score(y_test, y_pred, pos_label='spam')

    print("\n" + "=" * 65)
    print("               CONFUSION MATRIX")
    print("=" * 65)
    print("                     PREDICTED")
    print("                  Ham         Spam      Total")
    print(f"ACTUAL Ham   |   {tn:<8}    {fp:<8} | {tn + fp}")
    print(f"ACTUAL Spam  |   {fn:<8}    {tp:<8} | {fn + tp}")
    print("-" * 48)
    print(f"Total        |   {tn + fn:<8}    {fp + tp:<8} | {len(y_test)}")
    print("=" * 65)

    print("\n--- Detailed Interpretation ---")
    print(f" True Negatives (TN) [Ham as Ham]   : {tn} ({tn / (tn + fp) * 100:.1f}% of actual ham)")
    print(f" False Positives (FP)[Ham as Spam]  : {fp} ({fp / (tn + fp) * 100:.1f}% false alarms)")
    print(f" False Negatives (FN)[Spam as Ham]  : {fn} ({fn / (fn + tp) * 100:.1f}% spam slipped through!)")
    print(f" True Positives (TP) [Spam as Spam] : {tp} ({tp / (fn + tp) * 100:.1f}% of spam caught)")

    print("\n" + "=" * 65)
    print("               CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(y_test, y_pred, labels=labels, digits=4))

    print("=" * 65)
    print(f"Overall Accuracy         : {acc * 100:.2f}%")
    print(f"Spam Precision           : {prec_spam * 100:.2f}% (when predicted spam, is it really spam?)")
    print(f"Spam Recall (Catch rate) : {rec_spam * 100:.2f}% (what percentage of actual spam was caught?)")
    print(f"Spam F1-Score            : {f1_spam * 100:.2f}%")
    print("=" * 65)

    if show_graph:
        show_confusion_matrix_graph(cm, y_test, acc, prec_spam, rec_spam, f1_spam)

    return cm, acc, prec_spam, rec_spam, f1_spam


if __name__ == '__main__':
    evaluate_model(show_graph=True)
