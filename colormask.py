import cv2,os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint


data_path='F:\Face-Mask-Detection-master\dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary

print(label_dict)
print(categories)
print(labels)
img_size=100
data=[]
target=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            #Coverting the image into gray scale
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the gray scale into 100*100, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            target.append(label_dict[category])
            #appending the image and the label(categorized) into the list (dataset)

        except Exception as e:
            print('Exception:',e)
            #if any exception rasied, the exception will be printed here. And pass to the next image

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)

new_target=np_utils.to_categorical(target)
np.save('data',data)
np.save('target',new_target)
data = np.load('data.npy')
target = np.load('target.npy')

model=Sequential()
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)
Train on 2800 samples, validate on 701 samples
Epoch 1/20
2800/2800 [==============================] - 110s 39ms/step - loss: 0.6197 - accuracy: 0.6511 - val_loss: 0.5605 - val_accuracy: 0.6505
Epoch 2/20
2800/2800 [==============================] - 111s 40ms/step - loss: 0.4444 - accuracy: 0.7886 - val_loss: 0.3546 - val_accuracy: 0.8545
Epoch 3/20
2800/2800 [==============================] - 112s 40ms/step - loss: 0.3170 - accuracy: 0.8639 - val_loss: 0.2926 - val_accuracy: 0.8816
Epoch 4/20
2800/2800 [==============================] - 99s 35ms/step - loss: 0.2259 - accuracy: 0.9068 - val_loss: 0.2385 - val_accuracy: 0.9144
Epoch 5/20
2800/2800 [==============================] - 99s 35ms/step - loss: 0.1861 - accuracy: 0.9221 - val_loss: 0.2517 - val_accuracy: 0.9001
Epoch 6/20
2800/2800 [==============================] - 98s 35ms/step - loss: 0.1408 - accuracy: 0.9439 - val_loss: 0.2537 - val_accuracy: 0.9058
Epoch 7/20
2800/2800 [==============================] - 100s 36ms/step - loss: 0.1073 - accuracy: 0.9625 - val_loss: 0.2908 - val_accuracy: 0.9101
Epoch 8/20
2800/2800 [==============================] - 101s 36ms/step - loss: 0.0964 - accuracy: 0.9625 - val_loss: 0.2715 - val_accuracy: 0.9144
Epoch 9/20
2800/2800 [==============================] - 99s 35ms/step - loss: 0.0751 - accuracy: 0.9718 - val_loss: 0.2485 - val_accuracy: 0.9230
Epoch 10/20
2800/2800 [==============================] - 105s 37ms/step - loss: 0.0648 - accuracy: 0.9761 - val_loss: 0.2590 - val_accuracy: 0.9258
Epoch 11/20
2800/2800 [==============================] - 103s 37ms/step - loss: 0.0384 - accuracy: 0.9871 - val_loss: 0.2565 - val_accuracy: 0.9244
Epoch 12/20
2800/2800 [==============================] - 100s 36ms/step - loss: 0.0377 - accuracy: 0.9868 - val_loss: 0.2893 - val_accuracy: 0.9272
Epoch 13/20
2800/2800 [==============================] - 101s 36ms/step - loss: 0.0418 - accuracy: 0.9846 - val_loss: 0.3241 - val_accuracy: 0.9287
Epoch 14/20
2800/2800 [==============================] - 101s 36ms/step - loss: 0.0277 - accuracy: 0.9925 - val_loss: 0.2857 - val_accuracy: 0.9387
Epoch 15/20
2800/2800 [==============================] - 89s 32ms/step - loss: 0.0277 - accuracy: 0.9907 - val_loss: 0.2475 - val_accuracy: 0.9401
Epoch 16/20
2800/2800 [==============================] - 92s 33ms/step - loss: 0.0229 - accuracy: 0.9925 - val_loss: 0.3012 - val_accuracy: 0.9258
Epoch 17/20
2800/2800 [==============================] - 90s 32ms/step - loss: 0.0319 - accuracy: 0.9882 - val_loss: 0.2891 - val_accuracy: 0.9258
Epoch 18/20
2800/2800 [==============================] - 91s 32ms/step - loss: 0.0112 - accuracy: 0.9964 - val_loss: 0.3319 - val_accuracy: 0.9272
Epoch 19/20
2800/2800 [==============================] - 91s 32ms/step - loss: 0.0119 - accuracy: 0.9950 - val_loss: 0.3191 - val_accuracy: 0.9315
Epoch 20/20
2800/2800 [==============================] - 90s 32ms/step - loss: 0.0318 - accuracy: 0.9907 - val_loss: 0.3030 - val_accuracy: 0.9429
model.save('model-019.model')

from keras.models import load_model
model = load_model('E:\model-019.model')
face_clsfr=cv2.CascadeClassifier('E:\OpenCV-Python-Series-master\src\cascades\data\haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)
labels_dict={0:'NO mask',1:' MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
        
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
source.release()
