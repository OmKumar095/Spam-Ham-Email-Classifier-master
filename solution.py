import codecs
import email
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


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
    emails = pd.read_csv('DataSets/index/train.csv', sep=" ", header=None, names=['category', 'paths'])
    emails['body'] = ''
    print(emails.groupby("category").size())

    # -------------------------------------------------------------
    # Step 2: Read and clean email files
    # -------------------------------------------------------------
    print("\nReading and cleaning email contents...")
    for index in range(0, len(emails.paths)):
        file_path = "DataSets" + emails.paths[index][2:]
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
                cleaned_text = " ".join(soup.findAll(text=True))
                emails.loc[index, 'body'] = cleaned_text
        except FileNotFoundError:
            # Skip if specific file is not found
            continue

        if (index + 1) % 5000 == 0:
            print(f"Processed {index + 1}/{len(emails.paths)} emails...")

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
    # Step 4: Bag-of-Words with Stemming
    # -------------------------------------------------------------
    print("\nFitting StemmedCountVectorizer...")
    count_vect = StemmedCountVectorizer(stop_words='english').fit(temails['body'])
    X_train_counts = count_vect.transform(temails['body'])
    print(f"Bag of Words shape: {X_train_counts.shape}")

    # -------------------------------------------------------------
    # Step 5: TF-IDF Transformation
    # -------------------------------------------------------------
    print("\nFitting TF-IDF Transformer...")
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    print(f"TF-IDF shape: {X_train_tfidf.shape}")

    # -------------------------------------------------------------
    # Step 6: Train Multinomial Naive Bayes Classifier
    # -------------------------------------------------------------
    print("\nTraining Multinomial Naive Bayes...")
    clf = MultinomialNB().fit(X_train_tfidf, temails['category'])
    accuracy = clf.score(X_train_tfidf, temails['category'])
    print(f"Model Training Accuracy: {accuracy * 100:.2f}%")

    # -------------------------------------------------------------
    # Step 7: Prediction loop
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
