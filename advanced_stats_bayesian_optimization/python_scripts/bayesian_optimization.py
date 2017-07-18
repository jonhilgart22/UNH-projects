#! usr/bin/env python
import numpy as np
import seaborn as sns
import scipy as sp
import functools
import numpy as np
from scipy.stats import multivariate_normal
import scipy.stats as stats
import time
import scipy as scipy
import sys
import pandas as pd
import itertools
from numpy import linalg as la
__author__ = 'Jonathan Hilgart'


def BayesianOptimization(object):
    """A class to perform Bayesian Optimization."""

    def __init__(self, train_points_x, train_points_y, test_range, kernel = 'squarred_kernel'):
        """Define the parameters for the bayesian optimization.

        The train points should be x,y coordinate that you already know about your
        function"""
        if kernel == 'squarred_kernel':
            self.kernel = self.__squarred_kernel__
        elif kernel == 'matern':
            self.kernel = self.__matern_kernel__




    def __test_gaussian_process__(self, test_x, train_x, train_y_numbers ,
                                  kernel, return_cov = False, y_var=10,
                              return_sample = False, covariance_noise = 5e-5,
                              n_posteriors = 5):
        """Test one new point in the Gaussian process or an array of points
        Returns the mean, var from normal distribution from the sampled point.
        Return cov = True will return the full covariance matrix.

        If return_sample= True
        returns samples ( a vector) from the informed posterior and the uninformed prior distribution

        Covariance diagonal noise is used to help enforce positive definite matrices

        N_posteriors indicates the number of posterior functions to create when sampling"""


        # define the covariance matrices
        covariance_train_train = kernel(train_x,train_x,train=True)
        covariance_test_train  = kernel(test_x,train_x)
        covariance_test_test  = kernel(test_x,test_x)


        # Use cholskey decomposition to increase speed for calculating mean
        try :# First try,
            L_test_test = np.linalg.cholesky(covariance_test_test + covariance_noise*np.eye(len(covariance_test_test)))
            L_train_train = np.linalg.cholesky(covariance_train_train + covariance_noise*np.eye(len(covariance_train_train)))
            Lk = np.linalg.solve(L_train_train, covariance_test_train.T)
            mus = np.dot(Lk.T, np.linalg.solve(L_train_train, train_y_numbers)).reshape((len(test_x),))
            # Compute the standard deviation so we can plot it
            s2 = np.diag(covariance_test_test) - np.sum(Lk**2, axis=0)
            stdv = np.sqrt(abs(s2))


        # Full matrix calculation of mean and covariance, much slower than cholesky decomposition
        except Exception as e:
            print(e)#LinAlgError: # In case the covariance matrix is not positive definite
            # Find the near positive definite matrix to decompose
            decompose_train_train = nearestPD(covariance_train_train + covariance_noise*np.eye(len(train_x)))
            decompose_test_test = nearestPD(covariance_test_test + covariance_noise*np.eye(len(test_x)))

            # cholskey decomposition within the try except block
            L_train_train = np.linalg.cholesky(decompose_train_train  )
            L_test_test = np.linalg.cholesky(decompose_test_test  )
            Lk = np.linalg.solve(L_train_train, covariance_test_train.T)
            mus = np.dot(Lk.T, np.linalg.solve(L_train_train, train_y_numbers)).reshape((len(test_x),))
            # Compute the standard deviation so we can plot it
            s2 = np.diag(covariance_test_test) - np.sum(Lk**2, axis=0)
            stdv = np.sqrt(abs(s2))

    #         ##### FULL INVERSION ####
    #         mus = covariance_test_train  @ np.linalg.pinv(covariance_train_train) @ train_y_numbers
    #         s2 = covariance_test_test - covariance_test_train @ np.linalg.pinv(covariance_train_train ) \
    #                      @ covariance_test_train.T

        def sample_from_posterior(n_priors=3, n_post=5):
            """Draw samples from the prior distribution of the GP.
            len(test_x) is the number of samplese to draw.
            Resource: http://katbailey.github.io/post/gaussian-processes-for-dummies/.

            N-Posteriors / N-Priors tells the number of functions to samples from the dsitribution"""

            # Draw samples from the posterior at our test points.
            covariance_test_test  = kernel(test_x,test_x)
            try: # try inside sample from posterior function
                L = np.linalg.cholesky(covariance_test_test +  covariance_noise*np.eye(len(test_x))- np.dot(Lk.T, Lk))
            except Exception as e:
                # Find the neareset Positive Definite Matrix
                near_decompose = nearestPD(covariance_test_test +  covariance_noise*np.eye(len(test_x))- np.dot(Lk.T, Lk))
                L = np.linalg.cholesky(near_decompose.astype(float) ) # within posterior
            # sample from the posterior
            f_post = mus.reshape(-1,1) + np.dot(L, np.random.normal(size=(len(test_x),n_posteriors)))

            # Sample X sets of standard normals for our test points,
            # multiply them by the square root of the covariance matrix
            f_prior_uninformed = np.dot(L_test_test, np.random.normal(size=(len(test_x),n_priors)))
            # For the posterior, the columns are the vector for that function
            return (f_prior_uninformed, f_post)

        if return_cov == True:
            return y_pred_mean.ravel(), var_y_pred_diag.ravel(), var_y_pred

        if return_sample == True:
            f_prior, f_post = sample_from_posterior(n_post = n_posteriors)
            return mus.ravel(), s2.ravel(), f_prior, f_post
        else:
            return mus.ravel(), s2.ravel()

    def sample_from_function(self, y_function,
                        train_x, train_y_numbers, kernel, y_var,
                        testing_domain = range(-100,100,10), n = 20,
                        verbose=True):
    """Sample N times from the unknown function and for each time find the
    point that will have the highest expected improvement (find the maxima of the function).
    Verbose signifies if the function should print out the points where it is sampling

    Returns a generator of x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                list_of_expected_improvements, testing_domain, mus
          for improvements. Mus and Vars are the mean and var for each sampled point in the gaussian process.

    Starts off the search for expected improvement with a coarse search and then hones in on
    the domain the the highest expected improvement.

    Note - the y-function can EITHER by the actual y-function (for evaluation purposes, or an objective function
    (i.e. - RMSE))"""

    # STARTING POINTS
    if train_x.shape[1]>1: # Two dimensional
        # Random starting point
        try:
            start_point = np.array([testing_domain[np.random.choice(len(testing_domain)),:]])
        except:
            start_point = np.array([testing_domain[np.random.choice(len(testing_domain))]])
        #start at random point in the domain
        best_x = start_point
        try: # in case we are passing the actual function here
            best_y = y_function(start_point[0][0] , start_point[0][1])
        except: # we are passing the objective function (min RMSE)
            best_y = y_function(start_point[0][0],dimensions = 'two',hyperparameter_value_two = start_point[0][1])
        # for plotting the points sampled
        x_sampled_points = []
        y_sampled_points = []
        sampled_variance = []

    else: # One dimensional case
        start_point = np.random.choice(testing_domain)#start at random point in the domain
        best_x = start_point
        best_y = y_function(start_point) # initial best y value based on start point
        start_y = y_function(start_point)
        # for plotting the points sampled
        x_sampled_points = []
        y_sampled_points = []
        sampled_variance =[]


    for i in range(n):
        if i == 0:
            if train_x.shape[1]==1: ## one dimensional case
                testing_domain = np.array([testing_domain]).reshape(-1,1)



            # find the next x-point to sample
            mus, vars_,_,post = test_gaussian_process(testing_domain, train_x, train_y_numbers,
                                               y_var, kernel, return_sample = True, n_posteriors=30)


            sigmas_post = np.var(post,axis=1)
            mus_post = np.mean(post,axis=1)
            # get the expected values from the posterior distribution
            list_of_expected_improvements = expected_improvement(mus_post, sigmas_post ,best_y)

            max_improv_x_idx = np.argmax(list_of_expected_improvements)
            max_improv_x = testing_domain[max_improv_x_idx]

            if train_x.shape[1]==1:
                max_improv_y = y_function(max_improv_x)
            else: # Two D
                try: # see if we are passing in the actual function
                    max_improv_y = y_function(max_improv_x[0], max_improv_x[1])
                except: # we are passing the objective function in
                    max_improv_y = y_function(max_improv_x[0],dimensions = 'two', hyperparameter_value_two = max_improv_x[1])

            if max_improv_y > best_y: ## use to find out where to search next
                best_y = max_improv_y
                best_x = max_improv_x
            else:
                best_x = start_point


            sampled_variance.append(sigmas_post[max_improv_x_idx])

            # append the point to sample
            x_sampled_points.append(max_improv_x)
            y_sampled_points.append(max_improv_y)

            # append our new the newly sampled point to the training data
            train_x = np.vstack((train_x, max_improv_x))
            train_y_numbers = np.vstack((train_y_numbers, max_improv_y))
            yield x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                list_of_expected_improvements, testing_domain, mus


        else:
            # reformat testing domain to include more possible points
            if train_x.shape[1]==1:
                testing_domain = np.linspace(testing_domain[0],testing_domain[-1],
                                        num=1_000).reshape(-1,1)


            mus, vars_,_,post = test_gaussian_process(testing_domain, train_x, train_y_numbers,
                                               y_var, kernel, return_sample = True, n_posteriors=30)

            sigmas_post = np.var(post,axis=1)
            mus_post = np.mean(post,axis=1)

            # get the expected values from the posterior distribution
            list_of_expected_improvements = expected_improvement(mus_post, sigmas_post, best_y)


            # find the best x value
            max_improv_x_idx = np.argmax(list_of_expected_improvements)
            max_improv_x = testing_domain[max_improv_x_idx]

            if train_x.shape[1]==1:
                max_improv_y = y_function(max_improv_x)
            else: # Two D
                try: # see if we are passing in the actual function
                    max_improv_y = y_function(max_improv_x[0], max_improv_x[1])
                except: # we are passing the objective function in
                    max_improv_y = y_function(max_improv_x[0],dimensions = 'two', hyperparameter_value_two = max_improv_x[1])


            if max_improv_y > best_y: ## use to find out where to search next
                best_y = max_improv_y
                best_x = max_improv_x
            else:
                pass
            if verbose:
                print(f"Bayesian Optimization just sampled point = {max_improv_x}")
                print(f"Best x (Bayesian Optimization) = {best_x}, Best y = {best_y}")
                sampled_variance.append(sigmas_post[max_improv_x_idx])

                # append the point to sample
                x_sampled_points.append(max_improv_x)
                y_sampled_points.append(max_improv_y)

                # append our new the newly sampled point to the training data
                train_x = np.vstack((train_x, max_improv_x))
                train_y_numbers = np.vstack((train_y_numbers, max_improv_y))

                yield x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                    list_of_expected_improvements, testing_domain, mus

            else:

                sampled_variance.append(sigmas_post[max_improv_x_idx])

                # append the point to sample
                x_sampled_points.append(max_improv_x)
                y_sampled_points.append(max_improv_y)

                # append our new the newly sampled point to the training data
                train_x = np.vstack((train_x, max_improv_x))
                train_y_numbers = np.vstack((train_y_numbers, max_improv_y))

                yield x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                    list_of_expected_improvements, testing_domain, mus










    def nearestPD(self, A):
        """
        #https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

        Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

        def isPD(B):
            """Returns true when input is positive-definite, via Cholesky"""
            try:
                _ = la.cholesky(B)
                return True
            except la.LinAlgError:
                return False


        def __squarred_kernel__(a, b, param=2.0, train=False, train_noise = 5e-3, vertical_scale=50):
            """Calculated the squarred exponential kernel.
            Adds a noise term for the covariance of the training data
            Adjusting the param changes the difference where points will have a positive covariance
            Returns a covaraince Matrix.
            Vertical scale controls the vertical scale of the function"""
            if train == False:
                sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
                return vertical_scale*np.exp(-.5 * (1/param) * sqdist)

            else:
                noisy_observations = train_noise*np.eye(len(a))
                sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
                return vertical_scale*np.exp(-.5 * (1/param) * sqdist) + noisy_observations

        def __matern_kernel__(a,b,C_smoothness=3/2,train=False, train_noise = 5e-2):
        """The class of Matern kernels is a generalization of the RBF and the
        absolute exponential kernel parameterized by an additional parameter
        nu. The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for nu=0.5
        to the absolute exponential kernel. Important intermediate values are
        nu=1.5 (once differentiable functions) and nu=2.5 (twice differentiable
        functions).

        c_smoother = inf = RBF

        The train keyword is used to add noisy observations to the matrix"""
        if C_smoothness not in [1/2,3/2]:
            return "You choose an incorrect hyparameter, please choose either 1/2 or 3/2"
        matrix_norm = np.array([np.linalg.norm(a[i] - b,axis=(1)) for i in range(len(a))])
        if C_smoothness == 1/2:
            if train == True:
                max(np.var(a),np.var(b)) * np.exp(-matrix_norm) + np.eye(len(matrix_norm))*train_noise
            else:
                return max(np.var(a),np.var(b)) * np.exp(-matrix_norm)
        elif C_smoothness == 3/2:
            if train == True:
                return max(np.var(a),np.var(b))* (1 +np.sqrt(3)*matrix_norm)*np.exp(-np.sqrt(3)*matrix_norm) + np.eye(len(matrix_norm))*train_noise
            else:
                return max(np.var(a),np.var(b))* (1 +np.sqrt(3)*matrix_norm)*np.exp(-np.sqrt(3)*matrix_norm)
