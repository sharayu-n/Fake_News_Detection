import flask
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
import joblib
import os
import pickle
import newspaper
from newspaper import Article
import nltk
from urllib.parse import unquote  

nltk.download('punkt')

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = unquote(request.form['url'])
        article = Article(str(url))
        article.download()
        article.parse()
        article.nlp()
        news = article.summary

        pred = model.predict([news])
        return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))
    except Exception as e:
        return render_template('index.html', prediction_text='Error predicting news: {}'.format(str(e)))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
