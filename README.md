# Emotion Recognizer
ANN model written in Visual Studio's ML.Net that classifies the emotion expressed in a photograph of someones face. Before the model is able to classify the image, the image must be pre-processed as to extract the feature vector. This feature extraction is done by using a tool called DLibDotNet. With this tool we are able to map out specific points on the face in the image. With this map we calculate specific distances from certaian zones on the face. Ex: Distance from left eye to top of left eyebrow. With the values we get from these certain zones we are then able to create our feature vector and assign a specfic emotion to it. All the training data went through this pre-processing in order to be prepared correctly.

## Prerequisites

Before running this program ensure the following NuGet packages are installed. 
- DlibDotNet v19.21+
- Microsoft.ML v1.5+

Because of this programs utilisation of DlibDotNet, ensure that the **shape_predictor_68_face_landmarks.dat** file is included. This is essential for creating the landmark map on the faces.

For the program option to create a CSV file from the training data ensure that you either have the Image directory included in this repository or follow the same structure as this repository: Images > 'Data Set' > 'Emotion Labels' > 'Image File'

In the FeatureExtraction.cs file change the ``` _path ``` variable to the location of your directory

## Pre-Processing Visual Aid

Default Landmark Map            |  Landmark Map w/ Regions | Applied Map to Image 
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg" width="300" height="300">|  <img src="https://user-images.githubusercontent.com/71711553/115223073-52240c00-a103-11eb-8b3c-1f9a100dcdcf.png" width="300" height="300"> | <img src="https://user-images.githubusercontent.com/71711553/115224328-a4196180-a104-11eb-843e-0ea9e41f2a42.png" width="200" height="300">

## ANN Model Explained

This model takes our feature vector data and maps it to a key value pair. Then the data gets sent through a Neural Network that calculates the weights and biases for getting the intended classifications
