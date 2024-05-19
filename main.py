import cv2  # camera view
import numpy as np  # numerical operations
from keras.models import load_model  # deep learning
from keras.preprocessing import image
import time
import pygame
my_model = load_model("mymodel.h5")

cap = cv2.VideoCapture(0)  # default camera ID 0
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

pygame.init()
pygame.mixer.init()
def play_mask():
    pygame.mixer.music.load("AudioFiles/maskdetected.mp3")
    pygame.mixer.music.play()

def play_cover():
    pygame.mixer.music.load("AudioFiles/covermouth.mp3")
    pygame.mixer.music.play()

def play_no_mask():
    pygame.mixer.music.load("AudioFiles/nomask.mp3")
    pygame.mixer.music.play()

audio_interval = 5  # in seconds
last_audio_time = time.time()

while cap.isOpened():
    _, img = cap.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8)
    # default min neighbour is 4
    """"CHANGED MIN NEIGHBOURS"""
    # default scale is 1.1

    if len(face) > 0:
        for (x, y, w, h) in face:
            # this iterates over the coordinates of the rectangle obtained using Haar Cascade
            face_img = img[y:y + h, x:x + w]
            # the above extracts the ROI Region Of Interest (Face) which is the face inside the generated rectangle
            cv2.imwrite('temp.jpg', face_img)
            # here the ROI is saved in a JPEG file, to be used by our CNN model for predictioniction
            test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
            # here the image is loaded at 150x150 pixels (RGB)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            # here an extra dimension is added for batch sizes - number of images passed
            # the use of bigger batches allow utilisation of hardware accelerates like GPU, faster computations.
            # however if low latency is priority then fewer batch sizes ideal.
            prediction = my_model.predict(test_image)[0]
            # storing prediction from CNN model
            prediction = np.argmax(prediction)
            # used to find the highest probability float point of detection/none/incorrect

            if prediction == 2:
                # probability 2
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                # red rectangle (haar cascade face detection)
                cv2.putText(img, 'NO MASK DETECTED! :(', ((x + w) // 2, y + h - 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)
                # red text displayed (0 % coverage)
                if time.time() - last_audio_time >= audio_interval:
                    play_no_mask()
                    last_audio_time = time.time()
            elif prediction == 1:  # probability 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                # partial mask detected
                cv2.putText(img, 'Please cover your nose and mouth :/', ((x + w) // 2, y + h - 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # blue (50% coverage)
                if time.time() - last_audio_time >= audio_interval:
                    play_cover()
                    last_audio_time = time.time()
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # mask detected
                cv2.putText(img, 'MASK DETECTED :)', ((x + w) // 2, y + h - 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 3)
                if time.time() - last_audio_time >= audio_interval:
                    play_mask()
                    last_audio_time = time.time()
    cv2.imshow('Face Mask Detection System', img)
    # image and text is displayed
    if cv2.waitKey(1) == ord('q'):
        exit()
    # runs until Q pressed

cap.release()
# stops capturing
cv2.destroyAllWindows()
