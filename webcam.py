import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from playsound import playsound

def play_alarm_sound_function():
	while True:
		playsound.playsound('alarm-sound.mp3',True)
#Load the saved model
model = tf.keras.models.load_model('super_final2.h5')
video = cv2.VideoCapture(0)
Catagories=['NonFire','Fire']
alarmStatus = False
while True:
  _, frame = video.read()
  #Convert the captured frame into RGB
  im = Image.fromarray(frame, 'RGB')
  #Resizing into 300x300 because we trained the model with this image size.
  im = im.resize((300,300))
  img_array = image.img_to_array(im)
  img_array /= 255.0
  img_array = img_array.reshape(300,300,3)
  img_array = np.expand_dims(img_array,axis = 0) 
  probabilities = model.predict(img_array)[0]
  prediction = probabilities[0]
  
  if Catagories[int(prediction)]== "Fire":
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    print("fire!!!")
    playsound('alarm.mp3')
  cv2.imshow("Capturing", frame)
  key=cv2.waitKey(1)
  if key == ord('q'):
    break
video.release()
cv2.destroyAllWindows()
