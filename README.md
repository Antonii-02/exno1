l# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('/content/Data_set.csv')

numeric_cols = ['num_episodes', 'rating', 'current_overall_rank', 'lifetime_popularity_rank', 'watchers']

# IQR Detection
outlier_summary = []

for col in numeric_cols:
    data = df[col].dropna()

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    # Z-Score Detection (|z| > 3)
    z_scores = np.abs(stats.zscore(data))
    z_outliers_idx = data.index[z_scores > 3]

    outlier_summary.append({
        'Column': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Limit (IQR)': lower_bound,
        'Upper Limit (IQR)': upper_bound,
        'IQR Outliers Count': len(iqr_outliers),
        'Z-Score Outliers Count (|Z|>3)': len(z_outliers_idx),
        'Outlier Rows (IQR Indices)': iqr_outliers.index.tolist()
    })

outliers_df = pd.DataFrame(outlier_summary)
print(outliers_df[['Column', 'Lower Limit (IQR)', 'Upper Limit (IQR)', 'IQR Outliers Count', 'Z-Score Outliers Count (|Z|>3)']])

# Let's inspect specific outlier rows for key columns
for item in outlier_summary:
    if item['IQR Outliers Count'] > 0:
        col = item['Column']
        print(f"\n--- Outliers in '{col}' (IQR Method) ---")
        print(df.loc[item['Outlier Rows (IQR Indices)'], ['show_name', 'country', col]])

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightblue')
    axes[i].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('')

# Hide unused subplot (since 5 numeric cols in 2x3 grid)
fig.delaxes(axes[5])

plt.tight_layout()
# The plot will be displayed automatically by matplotlib in the notebook output

```
<img width="847" height="591" alt="Screenshot 2026-08-03 151600" src="https://github.com/user-attachments/assets/38d7d571-46b0-4ac9-acd5-e1c25c320e69" />
<img width="832" height="388" alt="Screenshot 2026-08-03 151628" src="https://github.com/user-attachments/assets/77902402-3f42-45e6-8e81-3970bfb4da71" />
<img width="845" height="507" alt="Screenshot 2026-08-03 151642" src="https://github.com/user-attachments/assets/5c9ed847-5f47-4dec-80e4-1234de084cf1" />

# Result
Executive SummaryThis script performs Outlier Detection and Exploratory Data Analysis (EDA) on 5 numerical features (num_episodes, rating, current_overall_rank, lifetime_popularity_rank, watchers) from a dataset (Data_set.csv). It identifies extreme values that could distort downstream statistical analysis or machine learning models.1. Key Processes ExecutedData Selection: Filters the dataset to focus exclusively on key numeric metrics.IQR Outlier Detection: Calculates $Q_1$ (25th percentile) and $Q_3$ (75th percentile) to set lower/upper boundaries ($Q_1 - 1.5 \times \text{IQR}$ and $Q_3 + 1.5 \times \text{IQR}$).Z-Score Outlier Detection: Computes standard deviation scores ($Z = \frac{X - \mu}{\sigma}$) and flags values exceeding $\vert{}Z\vert{} > 3$.Outlier Inspection: Summarizes outlier counts and outputs affected show names and countries for manual review.Data Visualization: Generates a 2x3 grid of boxplots using seaborn to visually display data distributions and extreme points.2. Why These Methods Were ImplementedIQR Method: A robust, non-parametric approach ideal for skewed features like watchers or num_episodes, as it isn't easily distorted by extreme values.Z-Score Method: A parametric approach best suited for roughly normal distributions to flag true statistical anomalies beyond 3 standard deviations.Boxplots: Provides immediate visual evidence of distribution skewness, spread, and isolated extreme points across all columns.3. Next StepsCapping / Winsorization: Replace extreme outliers with upper/lower threshold boundaries.Log Transformation: Apply $\log(x + 1)$ to handle highly right-skewed columns like watchers.
