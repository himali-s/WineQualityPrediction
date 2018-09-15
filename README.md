# WineQualityPrediction
Training and tuning a random forest for wine quality (as judged by wine snobs experts) based on traits like acidity, residual sugar, and alcohol concentration.. Using Scikit-Learn to build and tune a supervised learning model.
Using the random forest generator to train the model. 
Here's the list of all the features:

quality (target)
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
All of the features are numeric, which is convenient. However, they have some very different scales. So I have standardize the features. 
