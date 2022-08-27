# Maria de los Angeles Arista Huerta - A01369984

"""
With this dataset I want to predict the price of diamonds with respect to their caratage.

About the dataset:
    There are 53,940 diamonds in the dataset with 10 features 
    (carat, cut, color, clarity, depth, table, price, x, y, and z). 
    Most variables are numeric in nature, but the variables cut, color, and clarity are 
    ordered factor variables with the following levels.

    About the currency for the price column: it is Price ($)

    And About the columns x,y, and z they are diamond measurements as:
                                                x: length in mm 
                                                y: width in mm
                                                z: depth in mm """

#~~~~~~~~~~~~~~~~~~~~~~ LIBRARIES ~~~~~~~~~~~~~~~~~~~~~~
import numpy as np      # linear algebra
import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)

#~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~
def data():
    #Extract the data from excel
    #path = "c:\\Users\\angix\\Downloads\\Modulo3" - Directorio
    data = pd.read_csv('c:\\Users\\angix\\Downloads\\MachineLearning\\Diamonds_Prices2022.csv')            
    return data

def hyp(params, x): 
    # params - containing the parameter for element x of the sample
    # x - containing the values of the original samples
    
    # y = mx + b -> mx = theta_n * x_n 
    #                  params[i] * x[i]
    
    sum=0                          # Store the sumatory part of the prediction method
    for i in range (len(x)): 
        # Cycle will be repeated m times - equivalent to the number of samples *(len(x))* 
        sum =+ params[i]*x[i]
    return sum

def error(params, m, y): 
    # params - containing the parameter for element x of the sample
    # m - containing the values of the original samples
    # y - containing the values of the real result for each sample
    
    # Implement mean square error to know the amount of error in model
    # MSE = 1 / m * SUM(y_prima - y)^2
    # mse = cost  *  sum
    
    sum = 0                         # Store the sumatory part of the MSE formula 
    cost = (1 / m) 
    for i in range (len(m)):
        # Cycle will be repeated m times - equivalent to the number of samples *(len(m))* 
        y_pred = hyp (params, m[i]) # y_prima
        y_true = y[i]               # y
        print( "hyp =  %f  y = %f " % (y_pred,  y_true))   
        error = y_pred - y_true     # error = y_prima - y 
        sum =+ error**2             # error accumulated squared
    mse = cost * sum
    return mse
    
def optGD(params, x, y, a): #Gradient descent
    # params - containing the parameter for element x of the sample
    # x - containing the values of the original samples
    # y - containing the values of the real result for each sample
    # a - learning rate
    
    # Implement Gradient Descent formula for LR
    # GD    = thetaJ    - alpha / m * SUM(y_pred - y_real) * x
    #new[j] = params[j] - a * (1/len(x)) * sum 
    
    new = list(params)     
    for j in range(len(params)):        # Cycle to thetas
        # Cycle will be repeated j times - equivalent to the number of params
        sum =0                          # Store the sumatory part of the Gradient Descent formula
        for i in range(len(x)):         # Cycle to Sum
            # Cycle will be repeated x times - equivalent to the number of thetas (len(x))
            y_pred = hyp (params, x[i]) # y_prima
            y_true = y[i]               # y
            error = y_pred - y_true     # error = y_prima - y 
            sum =+ error * x[i][j]  
            new[j] = params[j] - a * (1/len(x)) * sum 
        return new

#~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~
#Arguments
params = [0,0]          #theta1 / bias
a = .03                 #learnig rate
epoch = 0               #counter of epochs
e = []                  #error

#Dataset
Data = data()
data = Data.drop(columns=['Unnamed: 0'], axis=1) #Delete the default index in the dataset
#print(data.head(3))#, print(data.info()), print(data.describe())

x = data[['carat']]     #Carat
y = data[['price']]     #Price (USD)
#print(x.head(3))#, print(x.info())
#print(y.head(3))#, print(y.info())

while True:  #  run gradient descent until local minima is reached
	old_params = list(params)
	params = optGD(params, x, y, a)	
	error(params, x, y) 
 
	print("~~~~~~~~~~~~~~\n",'Params: ', params)
 
	epoch =+ 1;
	if(old_params == params or epoch == 20):
    # When there is no further improvement with the optimization, 
    # is define the local minima
		print("~~~~~~~~~~~~~~\n","Samples: ", x)
		print("~~~~~~~~~~~~~~\n","Final params: ", params)
		break