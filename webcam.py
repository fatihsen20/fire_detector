import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

#Load the saved model
model = tf.keras.models.load_model('super_final.h5')
video = cv2.VideoCapture(0)
Catagories=['NonFire','Fire']
while True:
  _, frame = video.read()
  #Convert the captured frame into RGB
  im = Image.fromarray(frame, 'RGB')
  #Resizing into 224x224 because we trained the model with this image size.
  im = im.resize((300,300))
  img_array = image.img_to_array(im)
  # img_array = np.expand_dims(img_array, axis=0) / 255
  img_array /= 255.0
  img_array = img_array.reshape(300,300,3)
  img_array = np.expand_dims(img_array,axis = 0) 
  probabilities = model.predict(img_array)[0]
  prediction = probabilities[0]
  
  if Catagories[int(prediction)]== "Fire":
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    print("fire!!!")
  #if prediction is 0, which means there is fire in the frame.
  # if prediction <= 0.5:
  #   frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  #   print(prediction)
  cv2.imshow("Capturing", frame)
  key=cv2.waitKey(1)
  if key == ord('q'):
    break
video.release()
cv2.destroyAllWindows()

# test_image = image.load_img("C:/Users/Numan/Desktop/fire/indir.jpg",target_size=(300,300))
# test_image = image.img_to_array(test_image)
# test_image = test_image/255
# test_image = np.expand_dims(test_image,axis=0)

# #Predicting the class of the image
# result = model.predict(test_image)

# Catagories=['Fire','Smoke']

# image_show = PIL.Image.open("C:/Users/Numan/Desktop/fire/indir.jpg")


# plt.title(Catagories[int(result[0][0])])
# plt.imshow(image_show)