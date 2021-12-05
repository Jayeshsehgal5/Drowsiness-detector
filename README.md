# Drowsiness Detection 😴 🚗 

Driver drowsiness detection is a car safety technology which helps prevent accidents caused by the driver getting drowsy. Various studies have suggested that around 20% of all road accidents are fatigue-related, up to 50% on certain roads.
This project detects the eyes and yawn of the driver and generates an alert when the driver closes his/her eyes from more than 3-4 secs

## Description 📚
A computer vision system that can automatically detect driver drowsiness in a real-time video stream and then generate an alarm if the driver appears to be drowsy.

## Algorithm 💻
Faster RCNN is used with CNN architecture of InceptionV2.
The training is done on more than 3000 images for 4 classes - "Open Eyes" , "Close Eyes" , "Yawn" , "Not Yawning"
The training is done for 60000 steps per epoch. 


## Prediction with image

## Prediction with OpenCV

## Steps followed

- Image Dataset Collection
- Image Labelling
- Creation of labelmap 
- Conversion of protos into python files
- Coversion of images to XML files
- Conversion of XML files to tensor files
- Training the model 
- Conversion of ckpt files to infernce graph 
- Using inference graph and labelmap for prediction
- Using OpenCV for Camera access 
- Prediction with OpenCV

## Technology used
- Python
- Tensorflow/Keras
- Faster RCNN 
- Inception Architecture
- TFOD 
- Open CV
- Pycharm
