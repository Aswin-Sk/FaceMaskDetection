import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input 
from keras.applications import MobileNetV2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import LabelBinarizer
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report

INIT_LR = 1e-4
EPOCHS = 20
BS = 32
imagePaths=list(paths.list_images("dataset"))
data=[]
labels=[]
for imagePath in imagePaths:
    label=imagePath.split(os.path.sep)[-2] #masked_images/oo1.jpg
    image=preprocess_input(img_to_array(load_img(imagePath,target_size=(224,224))))
    data.append(image)
    labels.append(label)
data=np.array(data,dtype="float32")
labels=np.array(labels)
modellb=LabelBinarizer()
labels=modellb.fit_transform(labels)
labels=to_categorical(labels)
(xtest,xtrain,ytest,ytrain)=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42)

augmentedData=ImageDataGenerator(rotation_range=25,zoom_range=0.25,width_shift_range=0.15,height_shift_range=0.15,shear_range=0.23,horizontal_flip=True,fill_mode="nearest")

baseModel=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))
headModel=baseModel.output;
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(128,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation="softmax")(headModel)
model=Model(inputs=baseModel.input,outputs=headModel);
for layer in baseModel.layers:
    layer.trainable=False

optimizer = Adam(learning_rate=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
H = model.fit(augmentedData.flow(xtrain, ytrain, batch_size=BS), steps_per_epoch=len(xtrain) // BS, validation_data=(xtest, ytest), validation_steps=len(xtest) //BS, epochs=EPOCHS)


print("Evaluating ");
predIdxs=model.predict(xtest,batch_size=BS)
predIdxs=np.argmax(predIdxs,axis=1)
print(classification_report(ytest.argmax(axis=1),predIdxs,target_names=modellb.classes_))
model.save("mask.model",save_format="h5")
N=EPOCHS;
print(H.history)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["accuracy"])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["val_accuracy"])), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")




