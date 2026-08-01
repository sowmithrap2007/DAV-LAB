"""
4) A) DATA VISUALIZATION - NORMAL CURVES ON UCI DIABETES DATASET
AIM: To visualize the distribution of key numerical attributes in the UCI Diabetes dataset
using normal curves.
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load Dataset
uci_diabetes = pd.read_csv("uci_diabetes (3).csv")

# Plot Normal Curves for Glucose and BMI
plt.figure(figsize=(12, 5))

# Normal Curve for Glucose
plt.subplot(1, 2, 1)
sns.histplot(uci_diabetes["Glucose"], kde=True, stat="density", linewidth=0)
x = np.linspace(uci_diabetes["Glucose"].min(), uci_diabetes["Glucose"].max(), 100)
plt.plot(x, norm.pdf(x, uci_diabetes["Glucose"].mean(), uci_diabetes["Glucose"].std()), 'r')
plt.title("Normal Curve - Glucose")

# Normal Curve for BMI
plt.subplot(1, 2, 2)
sns.histplot(uci_diabetes["BMI"], kde=True, stat="density", linewidth=0)
x = np.linspace(uci_diabetes["BMI"].min(), uci_diabetes["BMI"].max(), 100)
plt.plot(x, norm.pdf(x, uci_diabetes["BMI"].mean(), uci_diabetes["BMI"].std()), 'r')
plt.title("Normal Curve - BMI")

plt.show()
