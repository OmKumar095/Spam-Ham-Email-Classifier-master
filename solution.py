import codecs
import email
import os
import re
import warnings
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.stem.snowball import SnowballStemmer
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

warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'DataSets')
INDEX_PATH = os.path.join(DATASETS_DIR, 'index', 'train.csv')


# Custom CountVectorizer with Snowball stemming
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def main():
    # -------------------------------------------------------------
    # Step 1: Load training email index
    # -------------------------------------------------------------
    print("Loading dataset index...")
    emails = pd.read_csv(INDEX_PATH, sep=" ", header=None, names=['category', 'paths'])
    print(emails.groupby("category").size())

    # -------------------------------------------------------------
    # Step 2: Read and clean email files
    # -------------------------------------------------------------
    print("\nReading and cleaning email contents...")
    bodies = []
    for index, rel_path_raw in enumerate(emails.paths):
        rel_path = rel_path_raw.lstrip('.').lstrip('/\\')
        file_path = os.path.join(DATASETS_DIR, rel_path)
        cleaned_text = ""
        try:
            with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as fp:
                email_text = email.message_from_file(fp)
                texts = ""
                if email_text.is_multipart():
                    for part in email_text.get_payload():
                        if part.get_content_maintype() == 'text':
                            payload = part.get_payload()
                            if isinstance(payload, str):
                                texts += payload
                else:
                    payload = email_text.get_payload()
                    if isinstance(payload, str):
                        texts += payload

                # Text cleaning
                texts = re.sub(r'\n', ' ', texts)
                texts = re.sub(r'_', '', texts)
                texts = re.sub(r'-', ' ', texts)
                texts = re.sub(r'/', ' ', texts)
                texts = re.sub(r':', ' ', texts)
                texts = re.sub(r'\$', '', texts)
                texts = re.sub(r'=', '', texts)
                texts = re.sub(r'<.*?>', '', texts)

                soup = BeautifulSoup(texts.lower(), 'html.parser')
                cleaned_text = " ".join(soup.stripped_strings)
        except FileNotFoundError:
            # Skip if specific file is not found
            pass

        bodies.append(cleaned_text)
        if (index + 1) % 1000 == 0 or (index + 1) == len(emails.paths):
            print(f"Processed {index + 1}/{len(emails.paths)} emails...")

    emails['body'] = bodies

    # -------------------------------------------------------------
    # Step 3: Filter empty samples and balance classes
    # -------------------------------------------------------------
    temails = emails[emails['body'] != ''][['category', 'body']]

    if len(temails) == 0:
        print("Warning: No emails were loaded. Check if email files exist in DataSets/data.")
        return

    sample_size = min(50000, len(temails[temails['category'] == 'spam']), len(temails[temails['category'] == 'ham']))
    # Sample with replacement as in notebook
    te_spam = temails.loc[temails['category'] == 'spam'].sample(sample_size, replace=True)
    te_ham = temails.loc[temails['category'] == 'ham'].sample(sample_size, replace=True)
    temails = pd.concat([te_spam, te_ham])

    print("\nBalanced dataset size:")
    print(temails.groupby('category').size())

    # -------------------------------------------------------------
    # Step 4: Stratified Train / Test Split
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        temails['body'],
        temails['category'],
        test_size=0.20,
        random_state=42,
        stratify=temails['category']
    )
    print(f"\nTrain set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # -------------------------------------------------------------
    # Step 5: Bag-of-Words with Stemming
    # -------------------------------------------------------------
    print("\nFitting StemmedCountVectorizer on train data...")
    count_vect = StemmedCountVectorizer(stop_words='english').fit(X_train)
    X_train_counts = count_vect.transform(X_train)
    print(f"Bag of Words shape: {X_train_counts.shape}")

    # -------------------------------------------------------------
    # Step 6: TF-IDF Transformation
    # -------------------------------------------------------------
    print("\nFitting TF-IDF Transformer...")
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    print(f"TF-IDF shape: {X_train_tfidf.shape}")

    # -------------------------------------------------------------
    # Step 7: Train Multinomial Naive Bayes Classifier
    # -------------------------------------------------------------
    print("\nTraining Multinomial Naive Bayes...")
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # -------------------------------------------------------------
    # Step 8: Evaluate with Confusion Matrix & Classification Report
    # -------------------------------------------------------------
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("             MODEL EVALUATION (TEST SET)")
    print("=" * 55)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix (labels=['ham', 'spam']):")
    print("                     PREDICTED")
    print("                  Ham         Spam")
    print(f"ACTUAL Ham   |   {tn:<8}    {fp:<8} (TN={tn}, FP={fp})")
    print(f"ACTUAL Spam  |   {fn:<8}    {tp:<8} (FN={fn}, TP={tp})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=['ham', 'spam'], digits=4))
    print("=" * 55)

    # Show confusion matrix graph window
    try:
        from evaluate import show_confusion_matrix_graph
        prec_spam = precision_score(y_test, y_pred, pos_label='spam')
        rec_spam = recall_score(y_test, y_pred, pos_label='spam')
        f1_spam = f1_score(y_test, y_pred, pos_label='spam')
        print("\nDisplaying Confusion Matrix Graph window...")
        show_confusion_matrix_graph(cm, y_test, acc, prec_spam, rec_spam, f1_spam)
    except Exception as e:
        print(f"Notice: Could not display graph window: {e}")

    # -------------------------------------------------------------
    # Step 9: Prediction loop
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Email Classifier Ready. Enter email text to classify (type 'exit' to quit):")
    print("=" * 50)
    while True:
        try:
            text = input("\nEnter Email body: ").strip().lower()
            if text in ('exit', 'quit'):
                break
            if not text:
                continue

            X_predict = count_vect.transform(np.array([text]))
            X_predict_tfidf = tfidf_transformer.transform(X_predict)
            prediction = clf.predict(X_predict_tfidf)
            print(f"--> Prediction: {prediction[0].upper()}")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == '__main__':
    main()
