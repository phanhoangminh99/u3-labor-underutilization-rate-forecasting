# U-3 Unemployment Rate Forecasting (U.S.)
A statistically grounded, forward-looking forecast of the U.S. U-3 unemployment rate using monthly macroeconomic indicators.

## TL;DR
- What: Forward-looking forecast of the U.S. U-3 unemployment rate using monthly macroeconomic and housing indicators
- Models: Interpretable linear regression with lagged unemployment and forward feature selection
- Key result: November 2024 unemployment forecast of 4.14%, closely matching the reported 4.2%
- Tech stack: Python, pandas, NumPy, statsmodels, custom regression utilities, matplotlib, seaborn

---

## Project Motivation
The U-3 unemployment rate is a headline economic indicator that shapes decisions for policymakers, businesses, and labor market analysts. While many forecasting approaches emphasize predictive power, they often rely on variables that are only observable after the fact or sacrifice interpretability for complexity.

This project was motivated by a practical question:

> *Using only information available at the time, how accurately can short-term unemployment be forecasted—and which economic signals truly matter?*

The goal was not to build a black-box model, but to develop a **transparent, statistically grounded forecasting framework** that balances accuracy, interpretability, and real-world constraints.

---

## Problem Statement
The U-3 Labor Underutilization Rate is a key economic indicator that directly influences policy decisions, labor market strategies, and business planning. Predicting this rate with accuracy is challenging due to the dynamic nature of the labor market, its dependence on macroeconomic factors (e.g., GDP trends), and changing social trends. Currently, stakeholders lack a timely, data-driven forecasting model capable of integrating multiple datasets and providing actionable insights for the upcoming Employment Situation Report in November 2024.

The main objectives include: 
- Predicting the U-3 Labor Underutilization Rate: Create an accurate, data-driven model to predict the November 2024 U-3 Labor Underutilization Rate, addressing the need for timely and reliable labor market forecasts.
- Integrating Diverse Datasets: Combine and preprocess data from multiple sources, including macroeconomic indicators as well as social indicators to capture the complexities of labor market dynamics.
- Identifying Key Predictive Factors: Analyze and prioritize the most impactful variables influencing the U-3 rate to improve model performance and interpretability.

---

## Data Overview

### Datasets
Integrate data from trusted sources, such as the U.S. Department of Labor Statistics, the Federal Reserve and U.S. Census Bureau.

### Timeframes 
Analyze data spanning 2004–2024, spanning 20 years, from July 2004 up to October 2024.

### Key Variables and Features

| Category                     | Variable                                           | Data Source                          | Timeframe           | Description                                                                           |
| ---------------------------- | -------------------------------------------------- | ------------------------------------ | ------------------- | ------------------------------------------------------------------------------------- |
| **Target**                   | Unemployment Rate (U-3)                            | Federal Reserve Economic Data (FRED) | Monthly (1948–2024) | Monthly U.S. unemployment rate used as the outcome variable.                          |
| **Consumption**              | Advance Monthly Sales for Retail and Food Services | U.S. Census Bureau                   | Monthly (1992–2024) | Early indicator of consumer demand across retail and food services.                   |
| **Inventories**              | Advance Retail Inventories                         | U.S. Census Bureau                   | Monthly (1992–2024) | Inventory levels held by U.S. retailers.                                              |
| **Inventories**              | Advance Wholesale Inventories                      | U.S. Census Bureau                   | Monthly (1992–2024) | Inventory levels held at the wholesale level.                                         |
| **Business Activity**        | Business Formation Statistics                      | U.S. Census Bureau                   | Monthly (2004–2024) | High-frequency data on new business formations in the U.S.                            |
| **Housing**                  | New Home Sales                                     | U.S. Census Bureau                   | Monthly (1963–2024) | National and regional data on new single-family homes sold.                           |
| **Housing**                  | New Residential Construction (Housing Starts)      | U.S. Census Bureau                   | Monthly (1963–2024) | Measures new residential construction activity.                                       |
| **Construction**             | Construction Spending                              | U.S. Census Bureau                   | Monthly             | Total U.S. construction expenditures across sectors.                                  |
| **Manufacturing**            | Manufacturers’ Shipments, Inventories, and Orders  | U.S. Census Bureau                   | Monthly             | Captures manufacturing output, demand, and supply-chain conditions.                   |
| **Macroeconomic**            | U.S. Monthly GDP History                           | Federal Reserve Economic Data (FRED) | Monthly (1992–2024) | Monthly nominal GDP estimates representing overall economic output.                   |
| **Trade**                    | International Trade in Goods and Services          | U.S. Census Bureau                   | Monthly             | Measures the U.S. trade balance and external economic conditions.                     |
| **Structural (Exploratory)** | NBER Recession Indicator                           | Federal Reserve Economic Data (FRED) | Monthly (1854–2024) | Binary indicator of recession periods (explored but excluded from final forecasting). |
| **Shock (Exploratory)**      | Confirmed COVID-19 Cases per Million People        | Our World in Data                    | Monthly (2020–2024) | Pandemic severity proxy used to capture structural labor-market shocks.               |


## Training Dataset

### Dataset Structure

| Type | Variable | Description |
|---|---|---|
| **Target** | Unemployment Rate (U-3) | Monthly U.S. unemployment rate (percentage). |
| **Continuous** | Advance Monthly Sales for Retail and Food Services | Early indicator of consumer demand. |
|  | Advance Retail Inventories | Inventory levels held by retailers. |
|  | Advance Wholesale Inventories | Inventory levels at the wholesale level. |
|  | Business Formation Statistics | High-frequency data on new business formations. |
|  | Construction Spending | Total construction expenditures across sectors. |
|  | International Trade in Goods and Services | Measures U.S. trade balance and external demand. |
|  | Manufacturers’ Shipments, Inventories, and Orders | Manufacturing output and supply-chain activity. |
|  | New Home Sales | Monthly data on new single-family homes sold. |
|  | New Residential Construction | Housing starts and construction activity. |
|  | U.S. Monthly GDP History | Monthly nominal GDP estimates. |
|  | Confirmed COVID-19 Cases per Million People | Pandemic severity proxy (exploratory). |
| **Categorical** | NBER Recession Indicator | Binary indicator of recession periods (exploratory). |

**Notes**:
- COVID-19 case data prior to 2020 was unrecorded and manually filled with zeros.
- All variables were aligned to a common monthly index.
- A lagged unemployment feature was engineered to capture temporal persistence.

---

**Note:** Data is limited to the United States. 


## Methodology

### Exploratory Analysis
- Correlation analysis revealed high multicollinearity among several Census Bureau indicators.
- Construction-related variables showed strong correlation with unemployment.
- Time-series plots highlighted the structural shock introduced by the COVID-19 pandemic :contentReference[oaicite:4]{index=4}.

### Time Series Modeling
- Autocorrelation analysis showed strong persistence in unemployment.
- A **Lag-1 unemployment rate** predictor was introduced, supporting an autoregressive modeling approach.

### Model Development
- Multiple Generalized Linear Models were evaluated:
  - Linear
  - Poisson
  - Gamma
  - Negative Binomial
- Forward feature selection was applied to:
  - Reduce overfitting
  - Improve interpretability
  - Address multicollinearity
  - Increase generalizability :contentReference[oaicite:5]{index=5}

### Feature Exclusions
- **Recession indicators** were excluded from final forecasting due to ex-post identification.
- **COVID-19 case counts** were excluded after confidence intervals included zero and raised stability concerns.
- Certain coefficients (e.g., early New Home Sales effects) were removed due to lack of intuitive interpretability.

---

## Final Model Specification
The final model includes:
- Lag-1 Unemployment Rate
- New Residential Construction
- New Home Sales
- Manufacturers’ Shipments, Inventories, and Orders
- Construction Spending
- Advance Monthly Sales for Retail and Food Services

These variables provided the best balance of statistical significance, interpretability, and forward-looking validity.

---

## Results

- Unemployment Model Prediction for November 2024: **4.1419%**
- FED Reported U3 Underutilization Rate, November 2024: **4.2%**


### Model Performance
- **R²:** 0.9948  
- **RMSE:** 0.1407  
- Predicted vs. actual values closely follow the y = x line, indicating strong fit.

<img width="406" height="310" alt="Screenshot 2026-01-12 at 9 13 47 PM" src="https://github.com/user-attachments/assets/ce2f1f57-5be6-4d7c-a762-22cfd8146812" />

---

## Model Diagnostics
Regression assumptions were explicitly evaluated:
- Residual autocorrelation (within confidence bands)
- Linearity and homoscedasticity (residuals vs. fitted)
- Normality (Q–Q plots, Shapiro–Wilk and Anderson–Darling tests)

Diagnostics supported the validity of the final linear model :contentReference[oaicite:7]{index=7}.

---

## Limitations
- The model is **predictive, not causal**.
- High R² is partially driven by unemployment persistence.
- Structural breaks remain difficult to forecast ex-ante.
- Results are specific to the U.S. labor market.

---

## How to Run

```bash
python src/AnalysisChallengeGroupII.py
```
