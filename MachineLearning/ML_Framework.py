# Maria de los Angeles Arista Huerta - A01369984

"""
With this dataset I want to predict the cut of diamonds 
with respect to their length, width and their depth.
Possible cuts:
            - Ideal
            - Premium
            - Very good
            - Good
            - Fair

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
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~
data = pd.read_csv('c:\\Users\\angix\\Downloads\\MachineLearning\\Diamonds_Prices2022.csv')
data = data.drop(columns=['Unnamed: 0'], axis=1) #Delete the default index in the dataset
#Reduce the dataframe to 530 values, the dataframe countaining 53944 samples
#Use the ten percent data of dataset to train the model
data = data.iloc[:530]

#print(data)
print(data['cut'].value_counts()) #posibles cortes 
