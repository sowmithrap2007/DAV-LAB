"""
4) C) Performing T-test on Diabetes Datasets
AIM: To perform a T-test on the UCI Diabetes and Pima Indians Diabetes datasets to compare
the means of numerical variables and determine statistical significance.
"""

# Import Libraries
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Load the Datasets
uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

# Select Relevant Numerical Columns
numerical_columns = ["Glucose", "BloodPressure", "BMI"]

# Perform Independent T-test
t_test_results = {}
for col in numerical_columns:
    t_stat, p_value = ttest_ind(uci_diabetes[col], pima_diabetes[col], equal_var=False)
    t_test_results[col] = {"T-statistic": t_stat, "P-value": p_value}

# Convert Results to DataFrame
t_test_df = pd.DataFrame(t_test_results).T

# Display Results
print("\nT-test Results:\n", t_test_df)
