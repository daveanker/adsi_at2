adsi_at2
==============================

Beer prediction using custom Pytorch neural network, and model deployment to Heroku using FastAPI.  
Predictions are based on user input for brewery name and a range of input scores for aroma, appearance, palate and taste. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── app                <- Folder for app deployment
    │   ├── main.py        <- Script for app deployment
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── checkpoints    <- Save model checkpoints to resume further training (if required)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── Dockerfile         <- List of commands to assemble Docker image
    │    
    ├── heroku.yml         <- Manifest to define Heroku app
    │    
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── sets.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── null.py
    │   │   └── performance.py
    │   │   └── pytorch.py       
    │   │
    │   └── utils          <- 'Helper' scripts
    │       └── misc.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
