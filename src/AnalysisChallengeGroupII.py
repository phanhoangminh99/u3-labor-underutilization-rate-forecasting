###########################################################
##                                                       ##
##  Finalized Code for the Analysis Challenge            ##
##                                                       ##
##  ADSP 31014 IP02 Statistical Models for Data Science  ##
##  Autumn Quarter 2024                                  ##
##                                                       ##
##  Date: 12-14-2024                                     ##
##  Group II: Samuel Martinez Koss                       ##
##            Kurt Fischer                               ##
##            Minh Phan                                  ##
##            Samuel Park                                ##
##                                                       ##
##  Work completed in a Google Colab Notebook            ##
##  Copied to Python file for grading                    ##
##                                                       ##
##  Thanks for a great quarter!                          ##
##                                                       ##
###########################################################

## --- SET-UP ---
import os
import sys
import datetime
import numpy
import pandas
import seaborn
import sys
from scipy.stats import (norm, shapiro, anderson, t, f, poisson, chi2, probplot)
from scipy.special import comb
from itertools import (chain, combinations)
from matplotlib.ticker import (MultipleLocator, StrMethodFormatter, FormatStrFormatter)
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.simplefilter('ignore', FutureWarning)

from google.colab import drive
drive.mount('/content/drive')
base_dir = '/content/drive/MyDrive/code + data'
sys.path.append(base_dir)
import Regression

pandas.options.display.float_format = '{:,.7f}'.format


## --- DATA LOADING ---
# UNRATE
unrateDF = pandas.read_csv(os.path.join(base_dir, 'UNRATE.csv'))
unrateDF = unrateDF.rename(columns={'DATE': 'Date', 'UNRATE': 'Unemployment Rate'})
unrateDF['Date'] = pandas.to_datetime(unrateDF['Date'])
unrateDF['Date'] = unrateDF['Date'].dt.strftime('%b-%Y')
unrateDF['Date'] = pandas.to_datetime(unrateDF['Date'], format='%b-%Y')

# Advance Monthly Sales for Retail and Food Services
amsrfsDF = pandas.read_excel(os.path.join(base_dir, 'Advance Monthly Sales for Retail and Food Services.xlsx'))
amsrfsDF = amsrfsDF.rename(columns={'Period': 'Date', 'Value': 'Advance Monthly Sales for Retail and Food Services'})
amsrfsDF['Date'] = pandas.to_datetime(amsrfsDF['Date'], format='%b-%Y')

# Advance Retail Inventories, MEASURED IN BILLIONS OF DOLLARS
ariDF = pandas.read_excel(os.path.join(base_dir, 'Advance Retail Inventories.xlsx'))
ariDF = ariDF.rename(columns={'Period': 'Date', 'Value': 'Advance Retail Inventories'})
ariDF['Date'] = pandas.to_datetime(ariDF['Date'], format='%b-%Y')

# Advance Wholesale Inventories, MEASURED IN BILLIONS OF DOLLARS
awiDF = pandas.read_excel(os.path.join(base_dir, 'Advance Wholesale Inventories.xlsx'))
awiDF = awiDF.rename(columns={'Period': 'Date', 'Value': 'Advance Wholesale Inventories'})
awiDF['Date'] = pandas.to_datetime(awiDF['Date'], format='%b-%Y')

# Business Formation Statistics, MEASURED IN THOUSANDS OF UNITS
bfsDF = pandas.read_excel(os.path.join(base_dir, 'Business Formation Statistics.xlsx'))
bfsDF = bfsDF.rename(columns={'Period': 'Date', 'Value': 'Business Formation Statistics'})
bfsDF['Date'] = pandas.to_datetime(bfsDF['Date'], format='%b-%Y')

# Construction Spending, MEASURED IN BILLIONS OF DOLLARS
csDF = pandas.read_excel(os.path.join(base_dir, 'Construction Spending.xlsx'))
csDF = csDF.rename(columns={'Period': 'Date', 'Value': 'Construction Spending'})
csDF['Date'] = pandas.to_datetime(csDF['Date'], format='%b-%Y')

# International Trade in Goods and Services, MEASURED IN BILLIONS OF DOLLARS
itgsDF = pandas.read_excel(os.path.join(base_dir, 'International Trade in Goods and Services.xlsx'))
itgsDF = itgsDF.rename(columns={'Period': 'Date', 'Value': 'International Trade in Goods and Services'})
itgsDF['Date'] = pandas.to_datetime(itgsDF['Date'], format='%b-%Y')

# Manufacturers' Shipments, Inventories, and Orders, MEASURED IN BILLIONS OF DOLLARS
msioDF = pandas.read_excel(os.path.join(base_dir, 'Manufacturers Shipments, Inventories, and Orders.xlsx'))
msioDF = msioDF.rename(columns={'Period': 'Date', 'Value': 'Manufacturers Shipments, Inventories, and Orders'})
msioDF['Date'] = pandas.to_datetime(msioDF['Date'], format='%b-%Y')

# New Home Sales, MEASURED IN THOUSANDS OF UNITS, "New Single Family Houses Sold"
nhsDF = pandas.read_excel(os.path.join(base_dir, 'New Home Sales.xlsx'))
nhsDF = nhsDF.rename(columns={'Period': 'Date', 'Value': 'New Home Sales'})
nhsDF['Date'] = pandas.to_datetime(nhsDF['Date'], format='%b-%Y')

# New Residential Construction, MEASURED IN THOUSANDS OF UNITS, "Housing Starts"
nrcDF = pandas.read_excel(os.path.join(base_dir, 'New Residential Construction.xlsx'))
nrcDF = nrcDF.rename(columns={'Period': 'Date', 'Value': 'New Residential Construction'})
nrcDF['Date'] = pandas.to_datetime(nrcDF['Date'], format='%b-%Y')

# US Monthly GDP History
usmgdphDF = pandas.read_excel(os.path.join(base_dir, 'US Monthly GDP History.xlsx'))
usmgdphDF = usmgdphDF.rename(columns={'DATE': 'Date', 'Monthly Nominal GDP Index': 'US Monthly GDP History'})
usmgdphDF['Date'] = pandas.to_datetime(usmgdphDF['Date'], format='%Y - %b')

# NBER Based Recession Indicators for the US -- ultimately excluded
nberbriusDF = pandas.read_csv(os.path.join(base_dir, 'USREC.csv'))
nberbriusDF = nberbriusDF.rename(columns={'DATE': 'Date', 'USREC': 'NBER Based Recession Indicators for the US'})
nberbriusDF['Date'] = pandas.to_datetime(nberbriusDF['Date'])
nberbriusDF['Date'] = nberbriusDF['Date'].dt.strftime('%b-%Y')
nberbriusDF['Date'] = pandas.to_datetime(nberbriusDF['Date'], format='%b-%Y')

recessions = {
    0.0: 'No Recession',
    1.0: 'Recession'
}
nberbriusDF['NBER Based Recession Indicators for the US'] = nberbriusDF['NBER Based Recession Indicators for the US'].map(recessions)

# Daily Confirmed COVID 19 Cases per Million People
ccovid19cmpDF = pandas.read_csv(os.path.join(base_dir, 'cumulative-confirmed-covid-19-cases-per-million-people.csv'))
ccovid19cmpDF = ccovid19cmpDF.drop(columns=['Entity'])
ccovid19cmpDF = ccovid19cmpDF.rename(columns={'Day': 'Date', 'Total confirmed cases of COVID-19 per million people': 'Confirmed COVID 19 Cases per Million People'})
ccovid19cmpDF['Date'] = pandas.to_datetime(ccovid19cmpDF['Date'], format='%Y-%m-%d')
ccovid19cmpDF['Confirmed COVID 19 Cases per Million People'] = ccovid19cmpDF['Confirmed COVID 19 Cases per Million People'].diff()

# Monthly Confirmed COVID 19 Cases per Million People -- ultimately excluded
Mccovid19cmpDF = ccovid19cmpDF.copy()
Mccovid19cmpDF = Mccovid19cmpDF.set_index('Date').resample('M').sum().reset_index()
Mccovid19cmpDF['Date'] = Mccovid19cmpDF['Date'] - pandas.offsets.MonthEnd(0) + pandas.offsets.MonthBegin(1)


## --- JOINING DATA ---
unrate_dataframes = [unrateDF, amsrfsDF, ariDF, awiDF, bfsDF, csDF, itgsDF, msioDF, nhsDF, nrcDF, usmgdphDF,
                     nberbriusDF, Mccovid19cmpDF]
merged_df = unrate_dataframes[0]
for df in unrate_dataframes[1:]:
    merged_df = pandas.merge(merged_df, df, on='Date', how='outer')
merged_df = merged_df.sort_values(by='Date')

merged_df['Confirmed COVID 19 Cases per Million People'] = merged_df['Confirmed COVID 19 Cases per Million People'].fillna(0.0)

# Include Lag 1 Variable
lag_merged_df = merged_df.copy()
lag_merged_df['Lag 1'] = lag_merged_df['Unemployment Rate'].shift(1)

print(lag_merged_df.describe())


## --- DATA ANALYSIS ---
# Displaying heatmap
plt.figure(figsize=(10, 8))
dataplot = seaborn.heatmap(merged_df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
plt.show()

# Uemployment Rate Distribution
plt.figure(figsize=(10, 10))
seaborn.histplot(merged_df['Unemployment Rate'], bins=30, kde=True)  # kde=True adds a kernel density estimate
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Unemployment Rate Time Series
unrateTSA = unrateDF.copy().drop(['Date'],axis=1)
unrateTSA = unrateTSA.to_numpy().T[0]
plt.figure(figsize=(20,10))
plt.title("Unemployment Rate")
plt.xlabel("Date Index")
plt.ylabel("Unemployment Rate")
plt.plot(unrateTSA)

# Unemployment Rate ACF Plot
plt.rc("figure", figsize=(20,10))
plt.figure(figsize=(20,10))
plot_acf(unrateTSA, lags=12)
plt.title("Autocorrelation Function Plot for Unemployment Rate")
plt.xlabel("Lag")
plt.ylabel("Correlation Coefficient")
plt.show()


## --- MODEL TRAINING SET-UP ---
most_recent_vals = {        # most up-to-date information for predicting November; last updated 12-11-2024
    'Date': datetime.date(2024,11,1),
    'Unemployment Rate': 4.2,
    'Advance Monthly Sales for Retail and Food Services': 718867.0,
    'Advance Retail Inventories': 824656.0,
    'Advance Wholesale Inventories': 905076.0,
    'Business Formation Statistics': 424656.0,
    'Construction Spending': 2173968.0,
    'International Trade in Goods and Services': -73836.0,
    'Manufacturers Shipments, Inventories, and Orders': 585376.0,
    'New Home Sales': 665.0,
    'New Residential Construction': 1425.0,
    'US Monthly GDP History': 29523.675539,
    'NBER Based Recession Indicators for the US': 'No Recession',
    'Confirmed COVID 19 Cases per Million People': 0.0,
    'Lag 1': 4.1
}

#######################################################
### MODEL TRAINING INITIALIZATION INCLUDING OCTOBER ###
#######################################################

cat_name = []#'NBER Based Recession Indicators for the US'] -- removed because unable to forecast with
int_name = ['Advance Monthly Sales for Retail and Food Services',
            'Advance Retail Inventories',
            'Advance Wholesale Inventories',
            'Business Formation Statistics',
            'Construction Spending',
            'International Trade in Goods and Services',
            'Manufacturers Shipments, Inventories, and Orders',
            'New Home Sales',
            'New Residential Construction',
            'US Monthly GDP History',
            #'Confirmed COVID 19 Cases per Million People', -- removed because of underlying code isses; CIs contained 0
            'Lag 1']

candidate_name = cat_name + int_name
candidate_count = len(candidate_name)

target_name = 'Unemployment Rate'

no_covid_df = lag_merged_df[lag_merged_df['Date'].dt.year != 2020]
no_covid_df.insert(0, 'Intercept', 1.0)

no_covid_df.loc[2039] = most_recent_vals
test_data = no_covid_df.copy().loc[2039]
test_data = test_data.fillna({
    "Intercept": 1.0
})
no_covid_df = no_covid_df.drop(2039, axis=0)

train_data = no_covid_df.copy()[['Intercept'] + candidate_name + [target_name]].dropna().reset_index(drop = True)

n_sample = train_data.shape[0]
y = train_data[target_name]

#######################################################
### MODEL TRAINING INITIALIZATION EXCLUDING OCTOBER ###
#######################################################

cat_name = []#'NBER Based Recession Indicators for the US']
int_name = ['Advance Monthly Sales for Retail and Food Services',
            'Advance Retail Inventories',
            'Advance Wholesale Inventories',
            'Business Formation Statistics',
            'Construction Spending',
            'International Trade in Goods and Services',
            'Manufacturers Shipments, Inventories, and Orders',
            'New Home Sales',
            'New Residential Construction',
            'US Monthly GDP History',
            #'Confirmed COVID 19 Cases per Million People',
            'Lag 1']

candidate_name = cat_name + int_name
candidate_count = len(candidate_name)

target_name = 'Unemployment Rate'

no_covid_df = lag_merged_df[lag_merged_df['Date'].dt.year != 2020]
no_covid_df.insert(0, 'Intercept', 1.0)

test_data = no_covid_df.loc[2038]
no_covid_df = no_covid_df.drop(2038, axis=0)

train_data = no_covid_df.copy()[['Intercept'] + candidate_name + [target_name]].dropna().reset_index(drop = True)

n_sample = train_data.shape[0]
y = train_data[target_name]


## --- FORWARD SELECTION ---
# The FSig is the sixth element in each row of the FTest
def takeFSig(s):
    return s[6]

enter_threshold = 0.1
q_show_diary = False
step_diary = []

var_in_model = ['Intercept']

# Step 0: Enter Intercept
X0 = train_data[var_in_model]
# X0.insert(0, 'Intercept', 1.0)
result_list = Regression.LinearRegression(X0, y)
m0 = len(result_list[5])

# residual_variance = result_list[2] and residual_df = result_list[3]
SSE0 = result_list[2] * result_list[3]

step_diary.append([0, 'Intercept', SSE0, m0] + 4 * [numpy.nan])

# Forward Selection Steps
for iStep in range(candidate_count):
    FTest = []
    for pred in candidate_name:
        X = train_data[[pred]]
        if (pred in cat_name):
            u = X[pred].astype('category')
            ufreq = u.value_counts(ascending = True)
            X[pred] = u.cat.reorder_categories(list(ufreq.index)).copy()
            X = pandas.get_dummies(X.astype('category'), dtype = float)
        X = X0.join(X)

        result_list = Regression.LinearRegression(X, y)
        m1 = len(result_list[5])
        SSE1 = result_list[2] * result_list[3]

        df_numer = m1 - m0
        df_denom = n_sample - m1
        if (df_numer > 0 and df_denom > 0):
            FStat = ((SSE0 - SSE1) / df_numer) / (SSE1 / df_denom)
            FSig = f.sf(FStat, df_numer, df_denom)
            FTest.append([pred, SSE1, m1, FStat, df_numer, df_denom, FSig])

    # Show F Test results for the current step
    if (q_show_diary):
        print('\n===== F Test Results for the Current Forward Step =====')
        print('Step Number: ', iStep)
        print('Step Diary:')
        print('[Variable Candidate | Residual Sum of Squares | N Non-Aliased Parameters | F Stat | F DF1 | F DF2 | F Sig]')
        for row in FTest:
            print(row)

    FTest.sort(key = takeFSig, reverse = False)
    FSig = takeFSig(FTest[0])
    if (FSig <= enter_threshold):
        enter_var = FTest[0][0]
        SSE0 = FTest[0][1]
        m0 = FTest[0][2]
        step_diary.append([iStep+1] + FTest[0])
        X = train_data[[enter_var]]
        if (enter_var in cat_name):
            X = pandas.get_dummies(X.astype('category'), dtype = float)
        X0 = X0.join(X)
        var_in_model.append(enter_var)
        candidate_name.remove(enter_var)
    else:
        break

forward_summary = pandas.DataFrame(step_diary, columns = ['Step', 'Variable Entered', 'Residual Sum of Squares', 'N Non-Aliased Parameters', 'F Stat', 'F DF1', 'F DF2', 'F Sig'])
result_list = Regression.LinearRegression(X0, y)

print(forward_summary)


## --- MODEL PERFORMANCE ---
print(result_list[0])

# Forecasting result
X1 = test_data
X1_date = X1['Date']
X1 = X1[var_in_model]
X1 = X1.reindex(X0.loc[0].index)
X1 = X1.fillna({
    "NBER Based Recession Indicators for the US_No Recession": 1.0,
    "NBER Based Recession Indicators for the US_Recession": 0.0
})
new_month_prediction = X1.dot(result_list[0]['Estimate'])

print("Prediction for month of:",X1_date,round(new_month_prediction,4),"%")

# Model attributes
y_prediction = X0.dot(result_list[0]['Estimate'])

covb = result_list[1]
residual_variance = result_list[2]
residual_df = result_list[3]
model_df = len(result_list[5])

# Calculate the generalized inverse from the covariance matrix of parameter estimates
XtX_ginv = covb / residual_variance

# Calculate the H matrix and the leverages
H_matrix = X0.dot(XtX_ginv).dot(X0.transpose())
leverage = pandas.Series(numpy.diag(H_matrix), index = H_matrix.index, name = 'leverage')

corr_y_prediction = Regression.PearsonCorrelation (y, y_prediction)
R_Square = numpy.square(corr_y_prediction)
y_residual = y - y_prediction
y_std_residual = y_residual / numpy.sqrt(residual_variance * (1.0 - leverage))
y_residual_deleted = y_residual / (1.0 - leverage)

print('Correlation Coefficient:',corr_y_prediction)
print('R^2:',R_Square)

print(y_prediction)


## --- MODEL EVALUATION ---
# R-Squared
r_squared = r2_score(y, y_prediction)
print(f"R-Squared: {r_squared:.4f}")

# Mean Squared Error
mse = mean_squared_error(y, y_prediction)
print(f"Mean Squared Error: {mse:.4f}")

# Compute RMSE from MSE
rmse = mse ** 0.5
print(f"Root Mean Squared Error: {rmse:.4f}")

# Residual Analysis
residuals = y - y_prediction

# Residual Summary Statistics
print("\nResidual Summary Statistics:")
print(residuals.describe())

# Normal Q-Q Plot of Residuals
plt.figure(figsize=(8, 6))
probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot of Residuals")
plt.show()

# Perform Shapiro-Wilk test
stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p_value:.4e}")

if p_value > 0.05:
    print("Residuals are normally distributed (fail to reject H0).")
else:
    print("Residuals are not normally distributed (reject H0).")

# Perform Anderson-Darling test
result = anderson(residuals, dist='norm')
print("Anderson-Darling Test Statistic:", result.statistic)
print("Critical Values:", result.critical_values)
print("Significance Levels:", result.significance_level)

# Interpret the test
alpha = 0.05  # significance level
critical_value = result.critical_values[result.significance_level == 5.0][0]
if result.statistic < critical_value:
    print("Residuals are normally distributed (fail to reject H0).")
else:
    print("Residuals are not normally distributed (reject H0).")

# Residuals vs Fitted Values Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_prediction, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# Leverage and Influence Diagnostics
# Leverage
plt.figure(figsize=(8, 6))
plt.scatter(leverage, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title("Residuals vs Leverage")
plt.xlabel("Leverage")
plt.ylabel("Residuals")
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 6))
seaborn.histplot(residuals, kde=True, bins=30)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Autocorrelation of Residuals
plt.figure(figsize=(8, 6))
plot_acf(residuals, lags=30, alpha=0.05)  # alpha=0.05 for 95% confidence bands
plt.title("Autocorrelation of Residuals", fontsize=14)
plt.xlabel("Lag (k)", fontsize=12)
plt.ylabel("Autocorrelation Coefficient", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(visible=True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Predicted vs Actual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y, y_prediction, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

# Ensure correct feature names in `all_features`
# Replace categorical feature names with their one-hot encoded counterparts
encoded_features = [col for col in X0.columns if col.startswith('NBER Based Recession Indicators for the US')]
all_features = [feature for feature in X0.columns if feature != 'Intercept']
all_features.extend(encoded_features)
all_features = list(set(all_features))

# Initialize Shapley Value dictionary
shapley_values = {feature: 0 for feature in all_features if feature != 'Intercept'}

# Calculate Shapley Values
for feature in shapley_values.keys():
    for k in range(1, len(all_features)):
        for subset in combinations([f for f in all_features if f != feature], k - 1):
            subset_with_feature = list(subset) + [feature]
            subset_without_feature = list(subset)

            # Add Intercept for each subset
            X_with = X0[['Intercept'] + subset_with_feature]
            X_without = X0[['Intercept'] + subset_without_feature]

            # Calculate SSE for subsets
            result_with = Regression.LinearRegression(X_with, y)
            result_without = Regression.LinearRegression(X_without, y)

            SSE_with = result_with[2] * result_with[3]
            SSE_without = result_without[2] * result_without[3]

            # Calculate contribution of the feature
            shapley_values[feature] += (1 / comb(len(all_features) - 1, k - 1)) * (SSE_without - SSE_with)

# Normalize Shapley Values
total_value = sum(shapley_values.values())
shapley_df = pandas.DataFrame(list(shapley_values.items()), columns=['Feature', 'Shapley Value'])
shapley_df['Percentage Contribution'] = 100 * shapley_df['Shapley Value'] / total_value

# Enhance Table Display
shapley_df = shapley_df.sort_values(by='Shapley Value', ascending=False).reset_index(drop=True)
shapley_df.index += 1  # Start index at 1 for table readability

# Format Table
shapley_df['Shapley Value'] = shapley_df['Shapley Value'].map('{:.4f}'.format)
shapley_df['Percentage Contribution'] = shapley_df['Percentage Contribution'].map('{:.2f}%'.format)

# Display the enhanced table
print(shapley_df.to_string(index=True))

