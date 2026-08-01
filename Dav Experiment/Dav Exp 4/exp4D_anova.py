"""
4) D) Perform ANOVA on Diabetes Datasets
AIM: To perform ANOVA (Analysis of Variance) on the UCI Diabetes and Pima Indians Diabetes
datasets to analyze differences between multiple group means.
"""

# Import Libraries
import pandas as pd
import numpy as np
from scipy.stats import f_oneway

# Load the Datasets
uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

# Select Relevant Numerical Columns
numerical_columns = ["Glucose", "BloodPressure", "BMI"]

# Perform One-Way ANOVA
anova_results = {}
for col in numerical_columns:
    f_stat, p_value = f_oneway(uci_diabetes[col], pima_diabetes[col])
    anova_results[col] = {"F-statistic": f_stat, "P-value": p_value}

# Convert Results to DataFrame
anova_df = pd.DataFrame(anova_results).T

# Display Results
print("\nANOVA Results:\n", anova_df)
