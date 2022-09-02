# Maria de los Angeles Arista Huerta - A01369984

"""
With this dataset I want to predict the caratage of diamonds 
with respect to their length, width and their depth.

About the dataset:
    There are 53,940 diamonds in the dataset with 10 features 
    (carat, cut, color, clarity, depth, table, price, x, y, and z). 
    Most variables are numeric in nature, but the variables cut, color, and clarity are 
    ordered factor variables with the following levels.

    And About the columns x,y, and z they are diamond measurements as:
                                                x: length in mm 
                                                y: width in mm
                                                z: depth in mm """

#~~~~~~~~~~~~~~~~~~~~~~ LIBRARIES ~~~~~~~~~~~~~~~~~~~~~~
import numpy as np              # linear algebra
import pandas as pd             # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Graphics

__error__=[];                   #error

#~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~
def hyp(params, x): 
    # params - containing the parameters thetas and bias
    # x - containing the values of the original samples
    # Implement the formula to predict 
    # y = mx + b -> mx = theta_n * x_n 
    #                  params[i] * x[i]
    
    sum=0                          # Store the sumatory part of the prediction method
    for i in range (len(params)): 
        # Cycle will be repeated x times - equivalent to the number of samples *(len(x))* 
        sum = sum + params[i]*x[i]
    return sum

def error(params, x, y): 
    # params - containing the parameters thetas and bias
    # x - containing the values of the original samples
    # y - containing the values of the real result for each sample
    
    # Implement mean square error to know the amount of error in model
    # MSE = 1 / m * SUM(y_prima - y)^2
    # mse = cost  *  sum
    global __error__
    sum = 0                         # Store the sumatory part of the MSE formula 
    for i in range (len(x)):
        # Cycle will be repeated x times - equivalent to the number of samples *(len(x))* 
        y_pred = hyp (params, x[i]) # y_prima
        y_true = y[i]               # y
        print( "hyp =  %f  y = %f " % (y_pred,  y_true))   
        error = y_pred - y_true     # error = y_prima - y 
        sum = sum + error**2        # error accumulated squared
    mse = sum/len(x)
    __error__.append(mse)           #array with all errors
    
def optGD(params, x, y, a): #Gradient descent
    # params - containing the parameters thetas and bias
    # x - containing the values of the original samples
    # y - containing the values of the real result for each sample
    # a - learning rate
    
    # Implement Gradient Descent formula for LR
    # GD    = thetaJ    - alpha / m * SUM(y_pred - y_real) * x
    # new[j] = params[j] - a * (1/len(x)) * sum 
    
    new = list(params)     
    for j in range(len(params)):        # Cycle to thetas
        # Cycle will be repeated j times - equivalent to the number of params
        sum =0                          # Store the sumatory part of the Gradient Descent formula
        for i in range(len(params)):    # Cycle to Sum
            # Cycle will be repeated x times - equivalent to the number of thetas (len(x))
            y_pred = hyp (params, x[i]) # y_prima
            y_true = y[i]               # y
            error = y_pred - y_true     # error = y_prima - y 
            sum = sum + error * x[i][j]  
        new[j] = params[j] - a * (1/len(x)) * sum 
    return new

#~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~
#Hyper parameters
params = [0,0,0]   #theta1 / bias
a = .2             #learnig rate
epoch = 0          #counter of epochs

#Dataset
data = pd.read_csv('c:\\Users\\angix\\Downloads\\MachineLearning\\Diamonds_Prices2022.csv')
data = data.drop(columns=['Unnamed: 0'], axis=1) #Delete the default index in the dataset
#Reduce the dataframe to 500 values, the dataframe countaining 53944 samples
data = data.iloc[:500]          
x = data[['x','y','z']].to_numpy()
x = np.c_[x, np.ones(len(x))]
y = data[['carat']].to_numpy()

while True:
    old_params = list(params)
    print("~~~~~~~~~~~~~~\n",'OldParams: ', params)
    params = optGD(params, x, y, a)
    error(params, x, y) 
    #print("~~~~~~~~~~~~~~\n",'Params: ', params)
    
    epoch = epoch + 1
    print(epoch)
    if(old_params == params or epoch == 50):
    # When there is no further improvement with the optimization, 
    # is define the local minima
        print("~~~~~~~~~~~~~~\n","Samples: ", x)
        print("~~~~~~~~~~~~~~\n","Final params: ", params)
        break

# Plot the array of errors to know its change
plt.plot(__error__)
plt.show()