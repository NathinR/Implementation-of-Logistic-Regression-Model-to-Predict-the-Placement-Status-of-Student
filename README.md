# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nathin R
RegisterNumber: 212222230090
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('/content/Placement_Data_Full_Class.csv')
dataset

dataset.head(5)
dataset.tail(5)

dataset = dataset.drop('sl_no', axis=1)

dataset.info
l=['gender','ssc_b','hsc_b']
dataset=dataset.drop(l,axis=1)
dataset

dataset.shape
dataset.info()
dataset["hsc_p"]=dataset["hsc_p"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()
dataset.dtypes

ataset["hsc_p"]=dataset["hsc_p"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset

dataset['hsc_s']=dataset['hsc_s'].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
X
Y
X.shape
Y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
dataset.head()

X_train.shape
Y_train.shape
Y_test.shape

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)
clf.score(X_test, Y_test)
dataset

clf.predict([[41,35,2,0,0,60,0,35,0]])
```

## Output:
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/1fb597c9-47d4-45d6-a931-85c75d1d2a3d)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/4ce250be-ad8b-48d5-8991-4e195d2189e2)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/b39d23d2-727f-460b-b3aa-77af62b6653b)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/dba62a22-63c0-4069-9f1f-b201700f1874)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/46f8be19-9598-4a28-a011-4f9d9fc9a891)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/181cd502-3883-444b-b852-ceddb8cbdf9e)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/29af271c-413e-436c-8244-620885b937bd)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/6405c788-8978-4e46-8b3c-8b315893dfbd)
![image](https://github.com/NathinR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679646/0f825574-6bb7-4f4c-b1d2-c05a28ea6753)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
