import cv2
import numpy as np
import pytesseract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from flask import Flask, render_template, request

app = Flask(__name__)
 
# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

# Define image pre - processing function


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Define sentiment analysis function


def get_sentiment(tweet):
    tweet_words = []
    for word in tweet.split():
        word = word.strip()  # Remove leading / trailing whitespace
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_words_no_newline = [word.replace('\n', '') for word in tweet_words]
    revised_tweet = " ".join(tweet_words_no_newline)

    encoded_tweet = tokenizer(revised_tweet, return_tensors='pt')
    output = model(** encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    result = {}
    for i in range(len(scores)):
        result[labels[i]] = scores[i]

    return result

# Define Flask routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from request
    image_file = request.files['image']

# Read image using OpenCV
    img_array = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Pre - process image
    preprocessed_img = preprocess_image(img)

# Run OCR on pre - processed image
    ocr_text = pytesseract.image_to_string(preprocessed_img)

# Get sentiment analysis results
    result = get_sentiment(ocr_text)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
