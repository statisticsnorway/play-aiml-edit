# play-aiml-edit

Project repository for grant on AIML4OS related to WP8 - data editing.

Created by:
sjentoft <susiejentoft@gmail.com>

---

Repository accompanying the article: 

Foss, A.H., Seierstad, A. and Mustad, R. 
Comparing Traditional Editing Methods and Machine Learning Approaches for Outlier Detection: A Simulation Study  

# Purpose 

This repository contains the code used to: 

- simulate accumulation errors in monthly turnover data, 

- construct features for machine-learning models, 

- calibrate and train outlier-detection methods, 

- evaluate model performance, 

- reproduce the results reported in the article.  

The study compares four approaches to detecting accumulation errors: 

- Change-from-previous-period rule 

- Hidiroglou–Berthelot (HB) method 

- Isolation Forest 

- LightGBM  

 

# Background 

The study is motivated by challenges in the Norwegian Retail Sales Statistics. 

A common reporting problem is the occurrence of accumulation errors, where respondents report year-to-date turnover instead of turnover for the current reference month. Such errors may remain undetected and can substantially affect published statistics. 

To evaluate alternative editing methods under controlled conditions, a simulation framework was developed using cleaned monthly turnover data from Statistics Norway. Simulated accumulation errors were then injected into the data and used as known ground truth. 

 

# Data Availability 

The original data are confidential establishment-level turnover data from Statistics Norway's Index of Wholesale and Retail Sales. These data cannot be publicly shared. Synthetic data is made for users without access to the confidential source data.  


## Input Data

Due to data confidentiality, the original dataset cannot be shared. However, the methods can be applied to any dataset with a similar structure. The required input is a panel dataset containing:

- **unit_id** (*string*): Unique identifier for each unit (e.g., firm or entity)
- **period** (*datetime*): Main time variable used in the analysis (e.g., 2018-01-01)
- **time_period** (*string or datetime*): String representation of `period`, used by the detection methods
- **value** (*float*): Variable of interest (e.g., turnover)
- **nace** (*string*): Industry classification code
- **nace2** (*string*): Industry classification at the 2-digit level
- **true_error** (*binary, optional*): Indicator of true errors (1 = error, 0 = non-error), used for evaluation
- **year_month** (*string*): Time variable formatted as year-month
 

# Simulation Design 

Accumulation errors were artificially introduced into otherwise cleaned turnover series. 

The simulation procedure: 

- A fixed proportion of establishments was randomly selected. 

- One calendar year was selected for each establishment. 

- A contiguous block of months was chosen to represent an accumulation-error episode. 

- Monthly turnover was replaced by cumulative turnover values over the episode. 

- A binary error indicator was created to provide ground-truth labels. 

This allows objective evaluation of all methods because the true error status is known.  

# Training and Evaluation Periods
 
A strictly temporal split was used throughout the study.
 
Calibration periods varied by method:
 
- **Change-from-Previous-Period**, **Hidiroglou–Berthelot (HB)**, and **Isolation Forest** were calibrated using data from **January 2018 to December 2023**.
- For **LightGBM**, data from **January 2018 to December 2022** were used for model training, while **January 2023 to December 2023** was used for hyperparameter tuning and model calibration.
 
The independent evaluation period covered **January 2024 to April 2025**.
 
This setup mimics an operational production environment and prevents information leakag 

# Repository Structure
 
The main workflow is executed in the following order within the `src` directory:
 
1. **data/**
- Load and prepare input data.
- Feature engineering and data preparation routines.
- Generate synthetic data for users without access to the confidential source data.
 
2. **accumulation_error/**
- Simulate accumulation errors.
- Create ground-truth error labels used for model evaluation.
 
3. **functions/**
- Utility functions used throughout the project.

 
4. **models/**
- Train, calibrate, and evaluate the machine-learning models:
- Isolation Forest
- LightGBM
 
5. **HB_and_accumulation_models/**
- Calibrate parameters for the traditional editing methods.
- Run the Hidiroglou–Berthelot (HB) method.
- Run the Change-from-Previous-Period method.
- Compare model and rule-based approaches.



# Methods 

Change-from-Previous-Period 

Observations are flagged when turnover increases beyond a calibrated threshold relative to the previous month. Candidate thresholds were calibrated on historical data.  

Hidiroglou–Berthelot 

The HB method uses period-to-period ratios combined with establishment size information to identify unusual changes. Parameters were calibrated separately by industry group and month pair.  

Isolation Forest 

Isolation Forest was used as an unsupervised anomaly-detection method. The contamination parameter was selected using historical calibration data.  

LightGBM 

LightGBM was trained as a supervised classification model using simulated error labels. 

Hyperparameters were optimized using Bayesian optimization (Hyperopt). Class imbalance was handled through class weighting.  

 

# Feature Engineering 

The machine-learning models use features derived from the turnover series. 

Feature groups include: 

- month and year indicators, 

- month-to-month changes, 

- year-over-year changes, 

- lagged turnover values, 

- rolling means, 

- rolling medians, 

- rolling standard deviations, 

- ratios to rolling averages, 

- z-scores, 

- coefficients of variation.  

These variables were designed to capture trends, seasonality, volatility and reporting irregularities. 
