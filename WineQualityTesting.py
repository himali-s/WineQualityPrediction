#Testing wine quality using random forest. 
#Using scikit learn for that
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data =  pd.read_csv(dataset_url,sep=';')
#print(data.head)
# making y as a required parameter in test data
y = data.quality
x = data.drop('quality' ,axis=1)
	# splitting data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 123, stratify = y)

	# Declare data processing steps
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators = 100))

	# Declare hyperparameters  to tune 
hyperparameters = {'randomforestregressor__max_features': ['auto','sqrt','log2'],'randomforestregressor__max_depth':[None,5,3,1]}

	# Tune model using cross-validation pipeline
	# cross-validation is used for kfold where the parameters of grid search are pipeline, hyperparamters and number of kfold u want
clf = GridSearchCV(pipeline,hyperparameters,cv=10)
clf.fit(x_train,y_train)

	# evaluating on test data 
	# printing results
	#Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)
 
# 9. Evaluate model pipeline on test data
pred = clf.predict(x_test)
print(r2_score(y_test,pred))
print(mean_squared_error(y_test,pred))

# save the model for future use
joblib.dump(clf,'rf_regressor.pkl')




