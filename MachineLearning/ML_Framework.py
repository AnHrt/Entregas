# Maria de los Angeles Arista Huerta - A01369984

"""
With this dataset I want to predict the caratage of diamonds 
with respect to their price, length, width and their depth.

About the dataset:
    There are 53,940 diamonds in the dataset with 10 features 
    (carat, cut, color, clarity, depth, table, price, x, y, and z). 
    Most variables are numeric in nature, but the variables cut, color, and clarity are 
    ordered factor variables with the following levels.

    And About the columns x,y, and z they are diamond measurements as:
                                                x: length in mm 
                                                y: width in mm
                                                z: depth in mm """

# ~~~~~~~~~~~~~~~~~~~~~~ LIBRARIES ~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~
data = pd.read_csv('c:\\Users\\angix\\Downloads\\MachineLearning\\Diamonds_Prices2022.csv')
#data = data.drop(columns=['Unnamed: 0','cut','clarity','color'], axis=1) #Delete the default index in the dataset
#Reduce the dataframe to 500 values, the dataframe countaining 53944 samples
data = data.iloc[:500]


x = data.drop(['Unnamed: 0','cut','clarity','color'], axis=1)
y = data['carat']

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,
                                                random_state=1)
model = LinearRegression()
model.fit(Xtrain, ytrain)
print('\nparameter value: ',model.coef_) 
print('\nbias value:      ',model.intercept_) 

ypred = model.predict(Xtest)

print('\n',model.score(x,y))
