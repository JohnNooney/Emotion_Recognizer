# Emotion Recognizer
ANN model written in Visual Studio's ML.Net that classifies the emotion expressed in a photograph of someones face. Before the model is able to classify the image, the image must be pre-processed as to extract the feature vector. This feature extraction is done by using a tool called DLibDotNet. With this tool we are able to map out specific points on the face in the image. With this map we calculate specific distances from certaian zones on the face. Ex: Distance from left eye to top of left eyebrow. With the values we get from these certain zones we are then able to create our feature vector and assign a specfic emotion to it. All the training data went through this pre-processing in order to be prepared correctly.

## Pre-Processing Visual Aid

[Default Map before face applied](!https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg)

## ANN Model Explained

This model takes our feature vector
