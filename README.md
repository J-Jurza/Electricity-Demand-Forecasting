# Electricity Demand Forecasting with Machine Learning and Time-Series Analysis Methods

## Introduction

Every day, electricity is consumed by individuals, households, and businesses around the world for a myriad of uses such as lighting, powering computers and machinery, heating, and cooling. More recently, electricity has also been crucial in refueling electric cars, supporting eco-friendlier commuting. However, alongside the vast consumption of electricity come significant challenges, particularly in accurately forecasting demand. Inaccurate forecasts can lead to excessive use of non-renewable energy sources and environmental harm or, conversely, underestimation can necessitate costly load shedding, potentially causing blackouts and outages. This report explores the application of machine learning models to accurately forecast electricity demand 30 minutes into the future, with the aim of mitigating these issues.

This project undertakes the challenge of forecasting electricity demand in Queensland (QLD), Australia, with an emphasis on precision and reliability. Accurate demand forecasting is pivotal for infrastructure planning, market operations, and policy formulation. By analyzing energy consumption trends and employing advanced machine learning models, this project seeks to mitigate the risks associated with demand prediction inaccuracies, such as unnecessary energy production and the resultant environmental impacts.

## Background and Significance

Given Australia's growing energy demand (and changes in its patterns), this research leverages historical data and predictive modeling to forecast future requirements. The evolving landscape of energy generation, marked by a significant shift towards renewable sources, necessitates accurate forecasting models to balance supply and demand effectively. This project aims to contribute to the energy sector's decision-making processes, ensuring sustainable and efficient energy management.

## Project Status

This project is a group capstone project for a UNSW Master of Data Science degree, signifying its importance in applying theoretical knowledge to solve real-world problems. The project was delivered with an industry-based final report outlining findings and suggestions for further research.

## Methodology

### Software and Tools

The analysis was conducted using Python and R programming languages. Python analyses were carried out in Jupyter Notebooks, leveraging Google Colab for its collaborative features and GPU support. Key Python libraries utilized included Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, Keras, TensorFlow, XGBoost, Statsmodels, and Hyperopt, covering a range of tasks from data cleaning to machine learning. R and RStudio facilitated EDA visualizations and report writing, employing packages like tidyverse and kableExtra. GitHub and Google Drive served as platforms for code repository and collaboration, respectively.

### Data Description and Preprocessing

The data consisted of 12 CSV files related to NSW, QLD, SA, and VIC, detailing forecast demand, temperature, and total demand. Preprocessing involved importing and merging these files into comprehensive DataFrames, standardising formats, and addressing inconsistencies. Efforts were made to ensure data integrity, such as correcting the DATETIME format and managing missing values through methods like linear interpolation or removal of biased data segments.

### Data Cleaning and Assumptions

Missing data were meticulously handled, with strategies tailored to preserve the dataset's accuracy without introducing bias. Assumptions included consistent DATETIME formatting and the relevance of temperature observation locations to their respective states. The analysis focused on data from 2016 onwards, considering the evolving nature of energy consumption patterns.

### Modelling Methods

Predictive models developed included Random Forest, XGBoost, and LSTM, chosen for their suitability in handling time-series data. The analysis was confined to QLD due to time constraints, employing 3-Fold Nested CV for model training, validation, and testing. Performance metrics such as MAE, MSE, RMSE, MAPE, and R-Squared facilitated comprehensive model comparison and evaluation against forecasts from the National Energy Market (NEM).


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


## Acknowledgements

Gratitude is extended to all contributors to this project, UNSW group members, supporting academic staff, and especially those providing the data and resources necessary for this analysis.
