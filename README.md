#  Wildlife Image Classification 
## Overview
This project builds an an advanced image classification system to identify wildlife species using Convolutional Neural network (CNN) and transfer learning (MobileNetV2) and also deploys using streamlit that users can upload the image and get the name of the species and also confidence scores.

## Objectives
This project demonstrates real-world Ml work
- data preprocessing
- Exploratory Data Analysis
- Training the CNN model
- Training the MobileNetV2 model
- Deploy a user-friendly ML application

## Models used
1. Custom CNN
- Conv2D → MaxPooling → Flatten → Dense
- Activation: ReLU, Softmax
- Built from scratch

2. MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Lightweight and efficient architecture
- Fine-tuned for wildlife classification
- Faster convergence and better generalization

## Dataset
- 12 Wildlife species dataset

## Technologies used
streamlit==1.55.0
tensorflow==2.21.0
numpy
pillow
gdown

## Features
- Drag or drop image for prediction
- Supports multiple image formats (JPG, JPEG,PNG,WEBP)
- Displays predicted class and confidence score
- clean and interactive UI

## Demo App
https:/06wildlifeimageclassification-xwzotl87qbtmwb4rfmy6gq.streamlit.app/

## Results
For Custom CNN: Good baseline performance ( accuracy 78.3 %)

For MobileNetV2: Higher accuracy ( accuracy almost 98%) Transfer learning significantly improved performance compared to training from scratch.

## Challenges
- Handling large model files (Github limit)
- Deployment issues on Streamlit Cloud
- Image format compatibility (.webp error)
I faces these error but I passed these challenges.

