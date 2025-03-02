import sys
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import thinkstats2
import thinkplot

# Add the path to your code files
sys.path.append(r'C:\Users\Adam\PycharmProjects\Term Project\code_files')
from nsfg import ReadFemPreg

# Load the data
df = ReadFemPreg()

# Updated list of variables for histograms
variables = ['agepreg', 'birthwgt_lb', 'birthwgt_oz', 'pregordr', 'babysex', 'gestasun_w']

# Create histograms
for var in variables:
    if var in df.columns:
        plt.figure()
        plt.hist(df[var].dropna(), bins=20, edgecolor='k', alpha=0.7)
        plt.title(f'Histogram of {var}')
        plt.xlabel(var)
        plt.ylabel('Frequency')
        plt.show()
    else:
        print(f"Column '{var}' not found in the DataFrame.")

# Display descriptive statistics
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
descriptive_stats = df[variables].describe()
print(descriptive_stats)

# PMF of Birth Weights
first_babies = df[df['pregordr'] == 1]
other_babies = df[df['pregordr'] != 1]
pmf_first = thinkstats2.Pmf(first_babies['birthwgt_lb'].dropna(), label='First Babies')
pmf_other = thinkstats2.Pmf(other_babies['birthwgt_lb'].dropna(), label='Others')

thinkplot.PrePlot(2)
thinkplot.Pmfs([pmf_first, pmf_other])
thinkplot.Config(xlabel='Birth Weight (lb)', ylabel='PMF', title='PMF of Birth Weights')
thinkplot.Show()

# CDF of Age of Pregnant Women
cdf_agepreg = thinkstats2.Cdf(df['agepreg'].dropna(), label='Age of Pregnant Women')

thinkplot.Cdf(cdf_agepreg)
thinkplot.Config(xlabel='Age of Pregnant Women', ylabel='CDF', title='CDF of Age of Pregnant Women')
thinkplot.Show()

# Scatter plot for agepreg vs. birthwgt_lb with regression line
df = df[['agepreg', 'birthwgt_lb']].dropna()
plt.figure(figsize=(10, 6))
plt.scatter(df['agepreg'], df['birthwgt_lb'], alpha=0.5, s=10)
slope, intercept, r_value, p_value, std_err = linregress(df['agepreg'], df['birthwgt_lb'])
x = np.array([df['agepreg'].min(), df['agepreg'].max()])
y = slope * x + intercept
plt.plot(x, y, color='red')
plt.xlabel('Age of Pregnant Women')
plt.ylabel('Birth Weight (lb)')
plt.title('Relationship Between Age of Pregnant Women and Birth Weight')
plt.show()

# Scatter plots and correlation analysis
df = ReadFemPreg()
df = df[['birthwgt_lb', 'agepreg', 'gestasun_w']].dropna()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['agepreg'], df['birthwgt_lb'], alpha=0.5, s=10)
plt.xlabel('Age of Pregnant Women')
plt.ylabel('Birth Weight (lb)')
plt.title('Birth Weight vs. Age')
plt.subplot(1, 2, 2)
plt.scatter(df['gestasun_w'], df['birthwgt_lb'], alpha=0.5, s=10)
plt.xlabel('Gestational Weeks')
plt.ylabel('Birth Weight (lb)')
plt.title('Birth Weight vs. Gestational Weeks')
plt.tight_layout()
plt.show()

cov_birthwgt_age = np.cov(df['birthwgt_lb'], df['agepreg'])[0, 1]
corr_birthwgt_age, _ = pearsonr(df['birthwgt_lb'], df['agepreg'])
print(f"Covariance (birth weight vs. age): {cov_birthwgt_age}")
print(f"Pearson’s correlation (birth weight vs. age): {corr_birthwgt_age}")

cov_birthwgt_gestasun_w = np.cov(df['birthwgt_lb'], df['gestasun_w'])[0, 1]
corr_birthwgt_gestasun_w, _ = pearsonr(df['birthwgt_lb'], df['gestasun_w'])
print(f"Covariance (birth weight vs. gestational weeks): {cov_birthwgt_gestasun_w}")
print(f"Pearson’s correlation (birth weight vs. gestational weeks): {corr_birthwgt_gestasun_w}")

# Simple Linear Regression
df = df[['agepreg', 'birthwgt_lb']].dropna()
X = sm.add_constant(df['agepreg'])
y = df['birthwgt_lb']
model = sm.OLS(y, X).fit()
print(model.summary())

# Multiple Regression Analysis
df = ReadFemPreg()
df = df[['agepreg', 'birthwgt_lb', 'gestasun_w']].dropna()
X = sm.add_constant(df[['agepreg', 'gestasun_w']])
y = df['birthwgt_lb']
model = sm.OLS(y, X).fit()
print(model.summary())

# Multiple Regression Analysis + Gender
df = ReadFemPreg()
df = df[['agepreg', 'birthwgt_lb', 'gestasun_w', 'babysex']].dropna()
X = sm.add_constant(df[['agepreg', 'gestasun_w', 'babysex']])
y = df['birthwgt_lb']
model = sm.OLS(y, X).fit()
print(model.summary())