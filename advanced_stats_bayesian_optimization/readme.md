This is a one-dimensional using a true function, two-dimensional using a true function, and two-dimensional using an objective function implementation of bayesian optimization.

- Start with the notebook **Overview_of_Intelligent_Bayesian_Optimization**, which explains
what bayesian optimization is.
- Then, within the notebooks folder you can view
**Gaussian_Process_Hyperparameter_Selection_1D_2D** which compares this implementation to
Grid Search and Random Grid Search over one and two dimensions.
- Next, the notebook
**Compare_IBO_to_Implementation** shows how my implementation performs compared to
the Bayesian_Optimization package
-Finally under python_scripts, you can view the *Intelligent Bayesian Optimization* class which contains a .fit(), .predict(), and .maximize() methods.

<h3 style="text-align: center;" markdown="1"> Below, you can view some of the results </h3>

#### 1) One dimensional  real function
![Alt text](images/1d_function.png?raw=true)

##### Expected Improvement for one-dimensional function
![Alt text](images/expected_improvement_1D.png?raw=true)

##### Results of optimization over one-dimensional function

![Alt text](images/1d_search_rand_grid_ibo.png?raw=true)
- Comparison between my implementation, random grid search, and grid search over a true one-dimensional function. 

#### 2) Two dimensional objective function
![Alt text](images/grad_boost_alcohol.png?raw=true)

##### Results of optimization over two-dimensional objective function
![Alt text](images/2d_search_grid_rand_hyp.png?raw=true)
- Comparison between my implementation, random grid search, and grid search over a true two-dimensional objective function. **Note**, the objective function here is *negative root mean squarred error* because Bayesian Optimization is a maximization technique (i.e. you can not minimize rmse).  

## Compare my implementation to the Bayesian_Optimization package
#### 3) Two dimensional real function
![Alt text](images/eggholder_function.png?raw=true)
##### Results of optimization over two-dimensional real function between my implementation and the Bayesian_Optimization package
![Alt text](images/Compare_IBO_to_Implementation_2d.png?raw=true)
- However, it must be noted that my implementation took ~ 124 seconds for 10 steps when searching ov ~6k parameters while the Bayesian_Optimization package took ~27 seconds.

# Resources
- Kevin Murphy: Machine Learning a Probabilistic Perspective
- Gaussian Processes : http://www.gaussianprocess.org/
