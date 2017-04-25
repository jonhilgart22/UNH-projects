Deep_Learning
==============================

Maximize revenue of NYC taxicabs with deep reinforcement learning (DQN).
This project was inspired by http://cs229.stanford.edu/proj2014/Jingshu%20Wang,%20Benjamin%20Lampert,%20Improving%20Taxi%20Revenue%20With%20Reinforcement%20Learning.pdf

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Not available on GitHub
    │  
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py   <- Uses a preprocessed .csv file to load into memory with auxiliary function (below)
    |   |   |__ auxiliary_functions.py <- functions to compute geohashes from latitudes. General data cleaning scripts
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions. Also, includes training data.
    │   │   ├── predict_model.py
    │   │   └── train_model.py    <- Train the RL algorithm
    |   |   |_ hyperparameter_optimization.py <- hyperparameter selection for the MLP model for our RL algorithm
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations. Additional images showing the loss and fares earned over time.
    │       └── plotting_geohashes.py   <- A script to see a heatmap of fares over time for January for NYC taxis
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
