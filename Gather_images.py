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

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == num_samples:
        break

 #   cv2.rectangle(frame, (10, 30), (310, 330), (0, 255, 0), 2)

    if start:
      #  roi = frame[100:500, 100:500]
        #time.sleep(1)
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, frame)
        count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 10), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
