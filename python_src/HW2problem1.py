import numpy as np
import os

np.random.seed(42)

# To plot pretty figures
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# Let's generate some simulated data.  This way, we'll know what the actual relationship between the inputs and 
# the outputs is and can see how good our estimate is. 
# 
# Suppose the actual relationship between the input $X$ and the output $Y$ is $Y=(X-3)(X-2)(X-1)+6$ with some 
# random noice $\epsilon$, a random variable from 
# a normal distribution with mean 0 and standard deviation 0.75. 


n = 100 # of examples in the simulated training dataset 

X = 4*np.random.rand(n, 1) # random.rand generates numbers in [0, 1) evenly
X.sort(axis=0)

f_X = abs(X -2) + 2
y = f_X + np.random.normal(0, 0.75, size=(n, 1))


plt.plot(X, y, "b.")    # blue dots are the relationship with "irreducible error" added
plt.plot(X, f_X, "g-")  # green curve is the actual relationship between X and Y

plt.xlabel("$x$")
plt.ylabel("$y$")

#plt.show()

# Let's find a ridge regression model to predict Y from X using a set of basis functions as inputs.  

# first set of basis functions, the monomials
from sklearn.preprocessing import PolynomialFeatures

p = 50 # degree of poly we are using to approximation the true relationship with
poly = PolynomialFeatures(degree=p)

X_b = poly.fit_transform(X)   # create matrix of monomials up to degree 10

lamb_performance = np.zeros((100))

for ii in range(1,10):
# next pick a value for lambda, the penalty parameter
        lamb = ii # Any value of lamb greater than 0

        # then use formula for the model weights beta which minimize training pMSE
        beta = np.linalg.inv(X_b.T.dot(X_b)+lamb*(np.identity(p+1))).dot(X_b.T).dot(y)
        beta  # the model weights

        # make a test dataset, also of size n

        X_test = 4*np.random.rand(n, 1)                              # random.rand generates numbers in [0, 1) evenly
        X_test.sort(axis=0) 
        y_test = (X_test-2)*(X_test-1)*(X_test-3)+6 + np.random.normal(0, 0.75, size=(n, 1))

        # compute test MSE
        y_pred = poly.fit_transform(X_test).dot(beta)
        test_MSE = (y_test-y_pred).T.dot(y_test-y_pred)/n

        print("test_MSE: " + str(test_MSE) + ", poly: " + str(p) + "," + " lamb: " + str(lamb))

        plt.plot(X, y_pred, "r-")    # red curve is the estimate of the relationship using degree 10 poly and lambda 1
        plt.plot(X, f_X, "g-")  # green curve is the actual relationship between X and Y

        plt.xlabel("$x$")
        plt.ylabel("$y$")

        #plt.show(block=True)


# The red curve looks pretty bumpy.  Try choosing other values for lambda and see if you can get a better estimate.  What happens to the estimate and test MSE when you increase lambda? decrease lambda?  Which value of lambda do you recommend be used in the estimate?  What is the value of the test MSE for your lambda?
