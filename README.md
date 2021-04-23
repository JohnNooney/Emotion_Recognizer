# Emotion Recognizer
This is an ANN model written in Visual Studio's ML.Net that classifies the emotion expressed in a photograph of someones face. In ML.Net a model is refered to as a transformer, in this project the SdcaMaximumEntropyMulticlassTrainer is utilized to tune the necessary weights for our model to make predicitons.

## Prerequisites

Before running this program ensure the following NuGet packages are installed. 
- DlibDotNet v19.21+
- Microsoft.ML v1.5+

Because of this programs utilisation of DlibDotNet, ensure that the **shape_predictor_68_face_landmarks.dat** file is included. This is essential for creating the landmark map on the faces.

For the program option to create a CSV file from the training data ensure that you either have the Image directory included in this repository or follow the same structure as this repository: Images > 'Data Set' > 'Emotion Labels' > 'Image File'

In the FeatureExtraction.cs file change the ``` _path ``` variable to the location of your directory

## Pre-Processing

Before the model is able to classify the image, the image must be pre-processed to extract the feature vector. This feature extraction is done by using a tool called DlibDotNet. With this tool we are able to map out specific points on the face in the image. With this map we calculate specific distances from certaian zones on the face. Ex: Distance from left eye to top of left eyebrow. With the values we get from these certain zones we are then able to create our feature vector and assign a specfic emotion to it. All the training data went through this pre-processing in order to be prepared correctly.

The specific process that was used to calculate each regions distance is as follows:

- **Left eyebrow**:
this will be a sum of the normalised distances between the left eyebrow landmarks and the inner point of the left eye (see Figure 2). Calculate each of the 4 normalised eyebrow distances by first subtracting point #40 from each left eyebrow point to produce 4 non-normalised  distances. Then divide each  of  the  non-normalised distances by the distance between points #40 and point #22 –this normalisesthe values according to the size of the specific face. Finally, sum all 4 normalised distances to produce just one “left eyebrow” feature and store it in a variable.

- **Right eyebrow**:
as above, but using the inner point of the right eye #43 and the right eyebrow points. Make sure to normalise the distances by diving with the corresponding right-side distance (coloured blue in Figure 2).

- **Left lip**:
as  above,  but  the  stationary  point  is  #34  and  the  distance  used  for normalisation is between #34 and #52. Use the 3 points on the left top part of the lip to construct the non-normalised distances: i.e. #49, #50, #51.

- **Right lip**:
as above, but use the 3 points on the right top part of the lip: i.e. #53, #54, #55.

- **Lip Width**:
this is just the distance between #49 and #55 divided by distance between #34 and #52 (for normalisation)

- **Lip Height**:
this is just the distance between #52 and #58 divided by distance between #34 and #52 (for normalisation

## Pre-Processing Visual Aid

Default Landmark Map (Fig. 1)           |  Landmark Map w/ Regions (Fig. 2) | Applied Map to Image (Fig. 3)
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg" width="300" height="300">|  <img src="https://user-images.githubusercontent.com/71711553/115223073-52240c00-a103-11eb-8b3c-1f9a100dcdcf.png" width="300" height="300"> | <img src="https://user-images.githubusercontent.com/71711553/115224328-a4196180-a104-11eb-843e-0ea9e41f2a42.png" width="200" height="300">

## ANN Model Explained

This model takes our feature vector data and maps it to a key value pair. Then the data gets sent through a training pipeline that adjusts the weights and biases until it reaches a satisfactory level of classification.

The algorthim used is that of Linear nature: the Stochastic dual coordinated ascent(SDCA) Maximum Entropy trainer. This algorthim makes multiclass classifications scalable, fast, cheap to train, and cheap to predict. They scale by the number of features and approximately by the size of the training data set. Also with this algorthim any hyperparameter tuning is not needed since SDCA algortithms yield good default performance.

The Maximum Entropy algorthim is a logistic regression algorithm at core. Meaning the main idea is to find a relationship between features and probability of a particular outcome. With logistic regression our prediciton will always be within the bounds of a certain area: 0% to 100%. Whereas with linear regression the prediciton is based on the overall range of inputs. 

With logistic regression, normally you will only be left with one prediciton but since maximum entropy is being used in a MultiClass classification manner it then becomes a multinomial logistic regression algorithm. So essentially for every class that exists there will be a probability associated with it and the class with the highest probablity will be the models prediction.

Linear vs Logisitic Regression         |  Multinominal Logisitic Regression 
:-------------------------:|:-------------------------:
![Logisitic Regression vs Linear Regression](https://www.machinelearningplus.com/wp-content/uploads/2017/09/linear_vs_logistic_regression.jpg)|  ![Logisitic Regression vs Linear Regression](https://www.statstest.com/wp-content/uploads/2020/05/Multinomial-Logistic-Regression-1-1024x676.jpg)

