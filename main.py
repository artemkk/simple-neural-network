import numpy as np

# Our neural network will model a single hidden layer with three inputs and one output. 
# In the network, we will be predicting the score of our exam based on the inputs of how many hours we studied 
# and how many hours we slept the day before. Our test score is the output.

# Import training data

# X = (hours studying, hours sleeping), y = score on test
x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data
y = np.array(([92], [86], [89]), dtype=float) # output
