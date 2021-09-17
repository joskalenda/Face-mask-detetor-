import cv2
import os
import sys
import time

try:
    #used to enter label name i.e. with mask/without mask
    label_name = sys.argv[1]
    #used to enter the number of image you want to collect
    num_samples = int(sys.argv[2])
except:
    print("Arguments missing.")
    exit(-1)

# path to directory where the image will be saved
IMG_SAVE_PATH = 'dataset'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists.".format(IMG_CLASS_PATH))
    print("All images gathered will be saved along with existing items in this folder")

cap = cv2.VideoCapture(0)

start = False
count = 0
