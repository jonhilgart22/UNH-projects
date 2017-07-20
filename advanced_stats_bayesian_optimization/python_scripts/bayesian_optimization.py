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
from scipy.stats import norm
from numpy import linalg as la
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import itertools
__author__ = 'Jonathan Hilgart'


class IBO(object):
    """
    IBO: Intelligent Bayesian OPtimization
    A class to perform Bayesian Optimization on a 1D or 2D domain.
    Can either have an objective function to maximize or a true function
    to maximize"""

    def __init__(self, kernel = 'squared_kernel'):
        """Define the parameters for the bayesian optimization.

        The train points should be x,y coordinate that you already know about your
        function"""
        if kernel == 'squared_kernel':
            self.kernel = self.__squared_kernel__
        elif kernel == 'matern':
            self.kernel = self.__matern_kernel__

    def fit(self, train_points_x, train_points_y,
            test_domain, train_y_func, y_func_type = 'real',
            samples = 10 , test_points_x = None, test_points_y = None,
            model_train_points_x = None, model_train_points_y = None,
            covariance_noise = 5e-5, n_posteriors = 30, kernel_params = None,
            model_obj = GradientBoostingRegressor,
            verbose = True):
        """Define the parameters for the GP.
        PARAMS:
        train_points_x, - x coordinates to train on
        train_points_y,  -  resulting output from the function, either objective or
        true function
        test_domain - the domain to test
        test_points_y - If using ab objective function, this is from the
            train test split data
        test_points_x = if using an objective function, this is from the
            train test split
        model - the model to fit for use with the objective function. Currently
            works with Gradient Boosting
        y_func_type - either the real function or the objective function.
            The objective function implemented in negative MSE (since BO is
            a maximization procedure)
        verbose = Whether to print out the points Bayesian OPtimization is
            picking
        train_y_func - This can either be an objective function or a true function
        kernel_params: dictionary of {'length':value} for squaredkernel
        model_train_points: the training points for the objective function
        """

        try:
            type(train_points_x).__module__ == np.__name__
            type(train_points_y).__module__ == np.__name__
        except Exception as e:
            print(e)
            return ' You need to input numpy types'
        # Store the training points
        self.train_points_x = train_points_x
        self.train_points_y = train_points_y
        self.test_domain = test_domain

        # setup the kernel parameters
        if kernel_params != None:
            self.squared_length = kernel_params['rbf_length']
        else:
            self.squared_length = None


        # Y func can either be an objective function, or the true underlying func.
        if y_func_type == 'real':
            self.train_y_func = train_y_func
        elif y_func_type == 'objective':
            if model_obj == None:
                return ' you need to pass in a model (GradientBoostingRegressor)'

            # Only if using an objective function, from the 'test' split
            self.test_points_x = test_points_x
            self.test_points_y = test_points_y
            self.model_train_points_x  = model_train_points_x
            self.model_train_points_y = model_train_points_y
            # model to train and fit
            self.model = model_obj
            self.train_y_func = self.hyperparam_choice_function



        # store the testing parameters
        self.covariance_noise = covariance_noise
        self.n_posteriors = n_posteriors
        self.samples = samples
        self.verbose = verbose


        if self.train_points_x.shape[1] ==1: # one dimension
            self.dimensions ='one'
        elif self.train_points_x.shape[1] ==2:
            self.dimensions = 'two'
        else:
            print('Either you entered more than two dimensions, \
                  or not a numpy array.')
            print(type(self.train_points_x))
        # create the generator
        self.bo_gen = self.__sample_from_function__(verbose=self.verbose)




    def predict(self):
        """returns x_sampled_points, y_sampled_points, best_x, best_y"""

        x_sampled_points, y_sampled_points, sampled_var, \
            best_x, best_y, improvements, domain, mus = next(self.bo_gen)

        return x_sampled_points, y_sampled_points, best_x, best_y

    def maximize(self, n_steps=10, verbose = None):
        """For the n_steps defined, find the best x and y coordinate
        and return them.
        Verbose controls whether to print out the points being sampled"""
        verbose_ = self.verbose
        self.samples = n_steps
        bo_gen = self.__sample_from_function__(verbose = verbose_)
        for _ in range(self.samples):
            x_sampled_points, y_sampled_points, sampled_var, \
                best_x, best_y, improvements, domain, mus = next(self.bo_gen)

        self.best_x = best_x
        self.best_y = best_y
        # return the best PARAMS
        return best_x, best_y



    def __test_gaussian_process__(self,  return_cov = False,
                              return_sample = False):
        """Test one new point in the Gaussian process or an array of points
        Returns the mu, variance, as well as the posterior vector.
        Improvements is the expected improvement for each potential test point.
        Domain, is the domain over which you are searching.

        Return cov = True will return the full covariance matrix.

        If return_sample= True
        returns samples ( a vector) from the
        informed posterior and the uninformed prior distribution

        Covariance diagonal noise is used to help enforce positive definite matrices

        """

        # Update the covaraince matrices
        self.covariance_train_train = self.kernel(self.train_points_x,
                                self.train_points_x, train=True)
        self.covariance_test_train  = self.kernel(self.test_domain,
                                                  self.train_points_x)
        self.covariance_test_test  = self.kernel(self.test_domain,
                                                 self.test_domain)


        # Use cholskey decomposition to increase speed for calculating mean
        try :# First try,
            L_test_test = np.linalg.cholesky(self.covariance_test_test + \
                    self.covariance_noise * np.eye(len(self.covariance_test_test)))
            L_train_train = np.linalg.cholesky(self.covariance_train_train + \
                    self.covariance_noise * np.eye(len(self.covariance_train_train)))
            Lk = np.linalg.solve(L_train_train, self.covariance_test_train.T)
            mus = np.dot(Lk.T, np.linalg.solve(L_train_train,
                                              self.train_points_y)).reshape(
                                    (len(self.test_domain),))
            # Compute the standard deviation so we can plot it
            s2 = np.diag(self.covariance_test_test) - np.sum(Lk**2, axis=0)
            stdv = np.sqrt(abs(s2))

        except Exception as e:
            print(e)#LinAlgError: # In case the covariance matrix is not positive definite
            # Find the near positive definite matrix to decompose
            decompose_train_train = self.nearestPD(
                self.covariance_train_train + self.covariance_noise * np.eye(
                    len(self.train_points_x)))
            decompose_test_test = self.nearestPD(
                self.covariance_test_test + self.covariance_noise * np.eye(
                    len(self.test_domain)))

            # cholskey decomposition on the nearest PD matrix
            L_train_train = np.linalg.cholesky(decompose_train_train)
            L_test_test = np.linalg.cholesky(decompose_test_test)
            Lk = np.linalg.solve(L_train_train, self.covariance_test_train.T)
            mus = np.dot(Lk.T, np.linalg.solve(L_train_train,
                                self.train_points_y)).reshape((len(self.test_domain)),)
            # Compute the standard deviation so we can plot it
            s2 = np.diag(self.covariance_test_test) - np.sum(Lk**2, axis=0)
            stdv = np.sqrt(abs(s2))

    #         ##### FULL INVERSION ####
    #         mus = covariance_test_train  @ np.linalg.pinv(covariance_train_train) @ train_y_numbers
    #         s2 = covariance_test_test - covariance_test_train @ np.linalg.pinv(covariance_train_train ) \
    #                      @ covariance_test_train.T

        def sample_from_posterior(n_priors=3):
            """Draw samples from the prior distribution of the GP.
            len(test_x) is the number of samplese to draw.
            Resource: http://katbailey.github.io/post/gaussian-processes-for-dummies/.

            N-Posteriors / N-Priors tells the number of functions to samples from the dsitribution"""


            try: # try inside sample from posterior function
                L = np.linalg.cholesky(self.covariance_test_test +
                    self.covariance_noise * np.eye(
                        len(self.test_domain))- np.dot(Lk.T, Lk))
            except Exception as e:
                print(e)
                # Find the neareset Positive Definite Matrix
                near_decompose = self.nearestPD(self.covariance_test_test +
                    self.covariance_noise * np.eye(
                        len(self.test_domain)) - np.dot(Lk.T, Lk))
                L = np.linalg.cholesky(near_decompose.astype(float) )
                # within posterior
            # sample from the posterior
            f_post = mus.reshape(-1,1) + np.dot(L, np.random.normal(
                size=(len(self.test_domain), self.n_posteriors)))

            # Sample X sets of standard normals for our test points,
            # multiply them by the square root of the covariance matrix
            f_prior_uninformed = np.dot(L_test_test,
                    np.random.normal(size=(len(self.test_domain), n_priors)))
            # For the posterior, the columns are the vector for that function
            return (f_prior_uninformed, f_post)

        if return_cov == True:
            return y_pred_mean.ravel(), var_y_pred_diag.ravel(), var_y_pred

        if return_sample == True:
            f_prior, f_post = sample_from_posterior()
            return mus.ravel(), s2.ravel(), f_prior, f_post
        else:
            return mus.ravel(), s2.ravel()


    def __sample_from_function__(self, verbose=None):
        """Sample N times from the unknown function and for each time find the
        point that will have the highest expected improvement (find the maxima of the function).
        Verbose signifies if the function should print out the points where it is sampling

        Returns a generator of x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                    list_of_expected_improvements, testing_domain, mus
              for improvements. Mus and Vars are the mean and var for each sampled point
               in the gaussian process.

        Starts off the search for expected improvement with a coarse search and then hones in on
        the domain the the highest expected improvement.

        Note - the y-function can EITHER by the actual y-function (for evaluation
        purposes, or an objective function
        (i.e. - RMSE))"""
        verbose = self.verbose


        # for plotting the points sampled
        x_sampled_points = []
        y_sampled_points = []
        best_x = self.train_points_x[np.argmax(self.train_points_y ),:]
        best_y =self.train_points_y [np.argmax(self.train_points_y ),:]



        for i in range(self.samples):
            if i == 0:
                if self.train_points_x .shape[1]==1: ## one dimensional case
                    testing_domain = np.array([self.test_domain]).reshape(-1,1)
                else:
                    testing_domain = self.test_domain

                # find the next x-point to sample
                mus, vars_, prior, post = self.__test_gaussian_process__(
                    return_sample = True)


                sigmas_post = np.var(post,axis=1)
                mus_post = np.mean(post,axis=1)
                # get the expected values from the posterior distribution
                list_of_expected_improvements = self.expected_improvement(
                    mus_post, sigmas_post ,best_y)

                max_improv_x_idx = np.argmax(np.array(
                    list_of_expected_improvements))
                #print(max_improv_x_idx,'max_improv_x_idx')
                max_improv_x = testing_domain[max_improv_x_idx]
                # don't resample the same point
                c = 1
                while max_improv_x in x_sampled_points:
                    if c == 1:
                        if self.train_points_x .shape[1]==1:
                            sorted_points_idx = np.argsort(list(np.array(
                                list_of_expected_improvements)))
                        else:
                            sorted_points_idx = np.argsort(list(np.array(
                                list_of_expected_improvements)),axis=0)
                    c+=1
                    max_improv_x_idx = int(sorted_points_idx[c])
                    max_improv_x = testing_domain[max_improv_x_idx]
                    # only wait until we've gon through half of the list
                    if c > round(len(list_of_expected_improvements)/2):
                        max_improv_x_idx = int(
                            np.argmax(list_of_expected_improvements))
                        max_improv_x = testing_domain[max_improv_x_idx]
                        break
                if self.train_points_x.shape[1]==1:
                    max_improv_y = self.train_y_func(max_improv_x)
                else: # Two D
                    try: # see if we are passing in the actual function
                        max_improv_y = self.train_y_func(
                            max_improv_x[0], max_improv_x[1])
                    except: # we are passing the objective function in
                        max_improv_y = self.train_y_func(
                            max_improv_x[0], dimensions = 'two',
                            hyperparameter_value_two = max_improv_x[1])
                if max_improv_y > best_y: ## use to find out where to search next
                    best_y = max_improv_y
                    best_x = max_improv_x
                if verbose:
                    print(f"Bayesian Optimization just sampled point = {best_x}")
                    print(f"Best x (Bayesian Optimization) = {best_x},\
                         Best y = {best_y}")
                    # append the point to sample
                    x_sampled_points.append(max_improv_x)
                    y_sampled_points.append(max_improv_y)
                    # append our new the newly sampled point to the training data
                    self.train_points_x = np.vstack((self.train_points_x,
                                                     max_improv_x))
                    self.train_points_y = np.vstack((self.train_points_y,
                                                     max_improv_y))

                    yield x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                        list_of_expected_improvements, testing_domain, mus

                else:
                    # append the point to sample
                    x_sampled_points.append(max_improv_x)
                    y_sampled_points.append(max_improv_y)

                    # append our new the newly sampled point to the training data
                    self.train_points_x = np.vstack((self.train_points_x, max_improv_x))
                    self.train_points_y = np.vstack((self.train_points_y, max_improv_y))

                    yield x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                        list_of_expected_improvements, testing_domain, mus


            else:

                if self.train_points_x.shape[1]==1:
                    testing_domain = np.array([testing_domain]).reshape(-1,1)
                else:
                    testing_domain = self.test_domain

                mus, vars_, prior, post = self.__test_gaussian_process__(
                        return_sample = True)

                igmas_post  = np.var(post,axis=1)
                mus_post = np.mean(post,axis=1)
                # get the expected values from the posterior distribution
                list_of_expected_improvements = self.expected_improvement(
                    mus_post, sigmas_post ,best_y)
                max_improv_x_idx = np.argmax(list_of_expected_improvements)
                max_improv_x = testing_domain[max_improv_x_idx]
                # don't resample the same point
                c = 1
                while max_improv_x in x_sampled_points:
                    if c == 1:
                        if self.train_points_x .shape[1]==1:
                            sorted_points_idx = np.argsort(list(np.array(
                                list_of_expected_improvements)))
                        else:
                            sorted_points_idx = np.argsort(list(np.array(
                                list_of_expected_improvements)),axis=0)
                    c+=1
                    max_improv_x_idx = int(sorted_points_idx[c])
                    max_improv_x = testing_domain[max_improv_x_idx]
                    # only wait until we've gon through half of the list
                    if c > round(len(list_of_expected_improvements)/2):
                        max_improv_x_idx = int(
                            np.argmax(list_of_expected_improvements))
                        max_improv_x = testing_domain[max_improv_x_idx]
                        break
                if self.train_points_x .shape[1]==1:
                    max_improv_y = self.train_y_func(max_improv_x)
                else: # Two D
                    try: # see if we are passing in the actual function
                        max_improv_y = self.train_y_func(
                            max_improv_x[0], max_improv_x[1])

                    except: # we are passing the objective function in
                        max_improv_y = self.train_y_func(
                            max_improv_x[0], dimensions = 'two',
                            hyperparameter_value_two = max_improv_x[1])

                if max_improv_y > best_y: ## use to find out where to search next
                    best_y = max_improv_y
                    best_x = max_improv_x
                if verbose:
                    print(f"Bayesian Optimization just sampled point = {max_improv_x}")
                    print(f"Best x (Bayesian Optimization) = {best_x}, Best y = {best_y}")
                    # append the point to sample
                    x_sampled_points.append(max_improv_x)
                    y_sampled_points.append(max_improv_y)


                    # append our new the newly sampled point to the training data
                    self.train_points_x = np.vstack((self.train_points_x, max_improv_x))
                    self.train_points_y = np.vstack((self.train_points_y, max_improv_y))

                    yield x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                        list_of_expected_improvements, testing_domain, mus

                else:
                    # append the point to sample
                    x_sampled_points.append(max_improv_x)
                    y_sampled_points.append(max_improv_y)

                    # append our new the newly sampled point to the training data
                    self.train_points_x = np.vstack((self.train_points_x, max_improv_x))
                    self.train_points_y = np.vstack((self.train_points_y, max_improv_y))

                    yield x_sampled_points, y_sampled_points, vars_, best_x, best_y, \
                        list_of_expected_improvements, testing_domain, mus




    def hyperparam_choice_function(self, hyperparameter_value,
                               dimensions = 'one', hyperparameter_value_two = None):
        """Returns the negative MSE of the input hyperparameter for the given
         hyperparameter.
        Used with GradientBoostingRegressor estimator currently
        If dimensions = one, then search n_estimators. if dimension equal
        two then search over n_estimators and max_depth"""
        #definethe model
        model = self.model
        # define the training points
        train_points_x = self.model_train_points_x
        train_points_y = self.model_train_points_y

        if self.dimensions == 'one':
            try:
                m = model(n_estimators= int(hyperparameter_value))
            except:
                 m = model(n_estimators= hyperparameter_value)
            m.fit(train_points_x, train_points_y)
            pred = m.predict(self.test_points_x )
            n_mse = self.root_mean_squared_error(self.test_points_y , pred)
            return n_mse
        elif self.dimensions =='two':
            try:
                m = model(n_estimators = int(hyperparameter_value),
                          max_depth = int(hyperparameter_value_two))
            except:
                m = model(n_estimators = hyperparameter_value,
                          max_depth = hyperparameter_value_two)
            m.fit(train_points_x, train_points_y)
            pred = m.predict(self.test_points_x)
            n_mse = self.root_mean_squared_error(self.test_points_y , pred)
            return n_mse
        else:
            return ' We do not support this number of dimensions yet'



    def root_mean_squared_error(self, actual, predicted, negative = True):
        """MSE of actual and predicted value.
        Negative turn the MSE negative to allow for
        maximization instead of minimization"""
        if negative == True:
            return -np.linalg.norm(actual - predicted)/np.sqrt(len(actual))
        else:
            return np.linalg.norm(actual - predicted)/np.sqrt(len(actual))

    def expected_improvement(self, mean_x, sigma_squared_x,
                             y_val_for_best_hyperparameters, normal_dist=None,
                              point_est = False):
        """Finds the expected improvement of a point give the current best point.
        If point_est = False, then computes the expected value on a vector
        from the posterior distribution.
        """

        with np.errstate(divide='ignore'): # in case sigma equals zero
            # Expected val for one point
            if point_est ==True:
                sigma_x = np.sqrt(sigma_squared_x) # get the standard deviation from the variance

                Z = (mean_x - y_val_for_best_hyperparameters) / sigma_x

                if round(sigma_x,8) == 0:
                    return 0
                else:
                    return (mean_x -
                            y_val_for_best_hyperparameters)*normal_dist.cdf(Z)+\
                            sigma_x*normal_dist.pdf(Z)

            else:
                # Sample from the posterior functions
                for _ in range(len(mean_x)):
                    list_of_improvements = []
                    m_s = []
                    for m, z, s in zip(mean_x, ((mean_x -y_val_for_best_hyperparameters)\
                                         / np.std(sigma_squared_x)),np.sqrt(sigma_squared_x) ):

                        list_of_improvements.append(((m-y_val_for_best_hyperparameters)*\
                                                     norm().cdf(z)\
                                                     +s * norm().pdf(z)))
                        m_s.append(m)

                    return list_of_improvements





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
        def isPD(B):
            """Returns true when input is positive-definite, via Cholesky"""
            try:
                _ = la.cholesky(B)
                return True
            except la.LinAlgError:
                return False



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
        while not self.isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3



    def __squared_kernel__(self, a, b, param=2.0, train=False,
                            train_noise = 5e-3, vertical_scale=1.5):
        """Calculated the squared exponential kernel.
        Adds a noise term for the covariance of the training data
        Adjusting the param changes the difference where points will have a positive covariance
        Returns a covaraince Matrix.
        Vertical scale controls the vertical scale of the function"""
        if self.squared_length != None:
            vertical_scale = self.squared_length

        if train == False:
            # ensure a and b are numpy arrays
            a = np.array(a)
            b = np.array(b)
            sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
            return vertical_scale*np.exp(-.5 * (1/param) * sqdist)

        else:
            # ensure a and b are numpy arrays
            a = np.array(a)
            b = np.array(b)
            noisy_observations = train_noise*np.eye(len(a))
            sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
            return vertical_scale*np.exp(-.5 * (1/param) * sqdist) + noisy_observations

    def __matern_kernel__(self, a,b,C_smoothness=3/2,train=False, train_noise = 5e-2):
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
                return max(np.var(a),np.var(b))* (1
                    + np.sqrt(3)*matrix_norm)*np.exp(-np.sqrt(3)*matrix_norm) \
                    + np.eye(len(matrix_norm))*train_noise
            else:
                return max(np.var(a),np.var(b))* (1 +np.sqrt(3) *
                                matrix_norm) * np.exp(-np.sqrt(3)*matrix_norm)
