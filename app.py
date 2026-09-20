import codecs
import email
import os
import pickle
import re
import traceback
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, 'DataSets', 'index', 'train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'app_model.pkl')

# Set NLTK path to locally bundled nltk_data directory
nltk_dir = os.path.join(BASE_DIR, 'nltk_data')
if nltk_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_dir)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

count_vect = None
tfidf_transformer = None
clf = None


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        try:
            stemmer = SnowballStemmer("english", ignore_stopwords=True)
        except Exception:
            stemmer = SnowballStemmer("english")
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def init_classifier():
    global count_vect, tfidf_transformer, clf

    if count_vect is not None and clf is not None:
        return

    if os.path.exists(MODEL_PATH):
        print(f"Loading cached model from {MODEL_PATH}...")
        with open(MODEL_PATH, 'rb') as f:
            count_vect, tfidf_transformer, clf = pickle.load(f)
        print("Classifier loaded from cache successfully!")
        return

    print("Initializing Classifier... Please Wait...")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Training dataset index not found at: {INDEX_PATH}")

    emails = pd.read_csv(INDEX_PATH, sep=" ", header=None, names=['category', 'paths'], nrows=5000)
    
    bodies = []
    print("Reading and cleaning emails...")
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

    emails['body'] = bodies
    temails = emails[emails['body'] != ''][['category', 'body']]

    te_spam = temails.loc[temails['category'] == 'spam']
    te_ham = temails.loc[temails['category'] == 'ham']
    sample_size = max(min(len(te_spam), len(te_ham)), 1000)
    te_spam = te_spam.sample(sample_size, replace=True)
    te_ham = te_ham.sample(sample_size, replace=True)
    temails = pd.concat([te_spam, te_ham])

    print("Fitting StemmedCountVectorizer...")
    count_vect = StemmedCountVectorizer(stop_words='english').fit(temails['body'])
    X_train_counts = count_vect.transform(temails['body'])

    print("Fitting TfidfTransformer...")
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    print("Training MultinomialNB...")
    clf = MultinomialNB().fit(X_train_tfidf, temails['category'])

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((count_vect, tfidf_transformer, clf), f)
    print("Saved model cache to disk.")


# Pre-initialize classifier at startup
try:
    init_classifier()
except Exception as e:
    print(f"Startup initialization error: {e}")


@app.route('/', methods=['POST', 'GET'])
def predict(name=None):
    global count_vect, tfidf_transformer, clf

    if request.method == 'POST':
        try:
            if clf is None or count_vect is None or tfidf_transformer is None:
                init_classifier()

            data = request.form
            text = data.get('content', '')
            if not text:
                return render_template('index.html', data={'status': False})

            X_predict = count_vect.transform(np.array([text]))
            X_predict_tfidf = tfidf_transformer.transform(X_predict)
            prediction = clf.predict(X_predict_tfidf)
            return render_template(
                'index.html',
                data={
                    'status': True,
                    'result': prediction[0].upper(),
                    'content': '"' + text + '"'
                }
            )
        except Exception as e:
            traceback.print_exc()
            return f"Prediction error: {e}", 500
    else:
        return render_template('index.html', data={'status': False}, name=name)


if __name__ == '__main__':
    init_classifier()
    print("Starting Flask web server on http://127.0.0.1:3000")
    app.run(host='0.0.0.0', port=3000, debug=False)