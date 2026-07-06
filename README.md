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

The original data are confidential establishment-level turnover data from Statistics Norway's Index of Wholesale and Retail Sales. These data cannot be publicly shared.  

The repository therefore contains: 

- code for all simulations and analyses, 

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

Researchers wishing to apply the methods should provide data with a similar structure and variable definitions.  

 

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

A strictly temporal split was used. 

Period, Purpose 

January 2018 – December 2023, Training and calibration 

January 2024 – April 2025, Independent evaluation 

This setup mimics operational production environments and avoids information leakage from future periods.  

 

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
