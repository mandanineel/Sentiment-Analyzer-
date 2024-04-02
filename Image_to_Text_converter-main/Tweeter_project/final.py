import cv2
import numpy as np
import pytesseract
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load image
img_path = 'tweet04.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f'Failed to load image from file: {img_path}')
else:
    # Pre-processing
    def preprocess_image(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        # _, thresh = cv2.threshold(gray, 210, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate the image
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Erode the image
        # eroded = cv2.erode(dilated, kernel, iterations=2)
        
        #  Gaussian blur
        # blurred = cv2.GaussianBlur(eroded, (1, 1), 0)
        
        return gray
    
if not isinstance(img, np.ndarray):
        print(f'img is not a valid numpy array: {img}')
else:
        # Apply pre-processing to image
        preprocessed_img = preprocess_image(img)

        # Run OCR on pre-processed image
        ocr_text = pytesseract.image_to_string(preprocessed_img)
        # print(ocr_text)
    
tweet = ocr_text + '😊'

tweet_words = []
for word in tweet.split():
    word = word.strip()  # Remove leading/trailing whitespace
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

# Create a new list with '\n' removed from each word
tweet_words_no_newline = [word.replace('\n', '') for word in tweet_words]
revised_tweet = " ".join(tweet_words_no_newline)
print(revised_tweet)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']
# sentiment analysis
encoded_tweet = tokenizer(revised_tweet, return_tensors='pt')
# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)