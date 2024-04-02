# import cv2
from PIL import Image
#import pytesseract
im_file = "tweet.jpg"
im = Image.open(im_file)
im.save("temp/tweet01.jpg") 
