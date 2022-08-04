# Tutorial Source: https://enlight.nyc/projects/neural-network

# A neural network does the following:
# 1) Takes inputs as a matrix (2D array of numbers)
# 2) Multiplies the input by a set of weights
# 3) Applies an activation function
# 4) Returns an output (a prediction)
# 5) Error is calculated by taking the difference from the desired output from the data and the predicted output. This creates our gradient descent, which we can use to alter the weights.
# 6) The weights are then altered slightly according to the error.
# 7) To train, this process is repeated 1,000+ times. The more the data is trained upon, the more accurate our outputs will be.

# Our neural network will model a single hidden layer with three inputs and one output. 
# In the network, we will be predicting the score of our exam based on the inputs of how many hours we studied 
# and how many hours we slept the day before. Our test score is the output.

# Import statements
import numpy as np

### NORMALIZATION OF TRAINING DATA ###

## Import training data

# X = (hours studying, hours sleeping)
x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data

# y = score on test
y = np.array(([92], [86], [89]), dtype=float) # output

## Scale training data (normalize against maxima to unveil true importance; feature normalization part of data preprocessing)

# Scale units
x_all = x_all/np.max(x_all, axis=0) # scaling input data
y = y/100 # scaling output data (max test score is 100)

##  Split data (into data for training and data for testing)
X = np.split(x_all, [3])[0] # training data (Data the neural network will use to develop it's weights and ergo predictive capabilities, includes inputs and outputs)
x_predicted = np.split(x_all, [3])[1] # testing data (Data neural network will implement it's capabilities on to provide a result value for some required query, so input only)

print(str(X))
print(str(x_predicted))


### FORWARD PROPAGATION ###
# Calculates estimate as result using randomized weights and given input variables

## Neural Network Class
class neural_network(object):  
    def __init__(self):
    #parameters    
        self.inputSize = 2    
        self.outputSize = 1    
        self.hiddenSize = 3