import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/Users/kirisanthiruchelvam/Library/CloudStorage/OneDrive-UniversityofGreenwich/Ai/HaarCasc_CNN_FM_Detector/FaceMaskDetectorCNN+HC/FMD_data/train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        '/Users/kirisanthiruchelvam/Library/CloudStorage/OneDrive-UniversityofGreenwich/Ai/HaarCasc_CNN_FM_Detector/FaceMaskDetectorCNN+HC/FMD_data/test',
        target_size=(150,150),
        batch_size=16,
        class_mode='categorical')
model_saved=model.fit_generator(
        training_set,
        epochs=2,
        validation_data=test_set,
        )
model.save('mymodel.h5',model_saved)


#To test for individual images
mymodel=load_model('mymodel.h5')

#test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
# test_image=image.load_img(r'C:/Users/karan/Desktop/Datasets/new_mask/test/incorrect_mask/01001_Mask_Mouth_Chin.jpg', target_size=(150,150,3))

test_image = image.load_img("path",target_size=(150,150,3))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
pred=mymodel.predict(test_image)[0]
print(np.argmax(pred))
