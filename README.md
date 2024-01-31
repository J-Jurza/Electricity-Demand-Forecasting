# Electricity Demand Forecasting

## Introduction

Every day, electricity is consumed by individuals, households, and businesses around the world for a myriad of uses such as lighting, powering computers and machinery, heating, and cooling. More recently, electricity has also been crucial in refueling electric cars, supporting eco-friendlier commuting. However, alongside the vast consumption of electricity come significant challenges, particularly in accurately forecasting demand. Inaccurate forecasts can lead to excessive use of non-renewable energy sources and environmental harm or, conversely, underestimation can necessitate costly load shedding, potentially causing blackouts and outages. This report explores the application of machine learning models to accurately forecast electricity demand 30 minutes into the future, with the aim of mitigating these issues.

## Project Status

This project is a capstone project for a Master of Data Science degree, signifying its importance in applying theoretical knowledge to solve real-world problems. The project is currently in the analysis phase, with ongoing efforts to refine the models and techniques used.

## Methodology

### Data Collection

Data has been collected from the four Australian states of NSW, QLD, VIC, and SA, encompassing half-hourly forecast electricity demand for the years 2016 to 2020, temperature data from 2010 to 2020, and total electricity demand data from the same period.

### Data Preparation

The dataset undergoes a thorough process of importation, cleaning, exploratory data analysis (EDA), and feature engineering/scaling to prepare it for machine learning applications.

### Model Development

We explore several machine learning models including LSTM, ARIMA, Random Forest, and XGBoost. Each model is initially trained with a set of baseline hyperparameters across 10 training runs to record the mean and 95% confidence interval for the RMSE accuracy metric. Hyperparameter tuning follows to enhance model performance.

### Model Evaluation

Models are evaluated based on their RMSE accuracy on both training and testing sets. The best-performing model will be determined by the lowest RMSE on the test set.

## Dependencies

The following Python packages are required to run the notebooks in this project:

```python
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import datetime as dt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error, accuracy_score
from statsmodels.graphics.tsaplots import plot_acf
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow import keras
from numpy import array

## Setup and Installation

Ensure you have Python (>= 3.7) installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/electricity-demand-forecasting.git
cd electricity-demand-forecasting
pip install -r requirements.txt
```

## Project Structure

The repository is structured as follows to facilitate clear understanding and easy navigation:

```bash
electricity-demand-forecasting/
│
├── src/                            # Source code for the project
│   ├── Data_Cleaning.ipynb         # Jupyter notebook for data cleaning and preparation
│   ├── LSTM_Model.ipynb            # Jupyter notebook for LSTM model development and training
│   ├── Load_Best_LSTM_Model.ipynb  # Jupyter notebook for loading and evaluating the best LSTM model
│   ├── RandomForest_Model.ipynb    # Jupyter notebook for Random Forest model development and training
│   ├── XGBoost_Model.ipynb         # Jupyter notebook for XGBoost model development and training
│
├── data/                           # Dataset and processed data
│
├── notebooks/                      # Additional Jupyter notebooks for EDA and model tuning
│
├── R_Visualisations.rmd            # R Markdown document for data visualisation
│
├── requirements.txt                # List of project dependencies
│
└── README.md                       # Project documentation
```

## Contributing

Contributions to improve the project are welcome. Feel free to open an issue or submit a pull request.


## Acknowledgements

Gratitude is extended to all contributors to this project, UNSW group members, supporting academic staff, and especially those providing the data and resources necessary for this analysis.
