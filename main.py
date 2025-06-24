#!/usr/bin/env python
# coding: utf-8

# 
# To optimize the strategy and inventory of a startup car dealership, it is crucial to anticipate the purchasing power and preferences of potential clients. This necessitates predicting the trends in clients' annual income and the general fluctuations in car prices over time. By understanding the trajectory of annual income, the dealership can tailor its inventory to match the financial capabilities of its target market, ensuring a lineup that aligns with current and future demand. Simultaneously, monitoring and forecasting price trends for cars sold allows for informed purchasing and pricing strategies, maximizing profitability and competitiveness. This dual-focus predictive analysis serves as a foundational element in crafting a business model that is responsive to market dynamics, customer affordability, and industry trends, ultimately enhancing the dealership's market position and financial success.

# ### Libraries

# In[2]:


import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from pandas import Timestamp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.models import ARIMA
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

from pmdarima.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from scipy.stats import mode
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# ### Reading CSV

# In[3]:


# Reading the csv
car_df = pd.read_csv("Car Sales.xlsx - car_data.csv")
crude_df = pd.read_csv('Stock Market Dataset.csv')
unemploy_df = pd.read_csv('UNRATE.csv')
yield_df = pd.read_csv('yield-curve-rates-1990-2023.csv.csv')


# ### EDA (Cars)

# In[4]:


# Checking the type of data for each column
car_df.info()


# In[5]:


# Describing the basic stats for numerical columns
car_df.describe().T


# In[6]:


# Unique values per column
car_df.nunique()


# In[7]:


# Checking for Null Values
car_df.isnull().sum()


# ### Categorical EDA (Cars)

# In[8]:


# Unique Values in categorical columns
cat_columns = ['Gender', 'Company', 'Engine', 'Transmission', 
               'Color', 'Body Style', 'Dealer_Region']

for col in cat_columns:
    print(f'\n>> Unique categories in {col}:\n', sorted(car_df[col].unique()))


# In[9]:


# Unique Car company and Body Style

unique_company = car_df['Company'].unique()
unique_car_style = car_df['Body Style'].unique()

print("Unique Car Company: \n", unique_company)
print("\nUnique Car Body Style: \n", unique_car_style)


# In[10]:


# Checking how many cars are sold by company
car_counts = car_df['Company'].value_counts()
print(car_counts)


# In[11]:


# Visualizing Bar Chart for Car Sales per Company

car_counts = car_counts.sort_values()

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(car_counts.index, car_counts.values)

# Add bar labels
ax.bar_label(bars, padding=5)

ax.set_xlabel('Quantity')
ax.set_title('Distribution of Company by Car Sales')
plt.grid(visible=True)  # Set visible to True to show the grid
plt.show()


# In[12]:


# Checking which body style is the most popular
car_body_style = car_df['Body Style'].value_counts()
print(car_body_style)


# In[13]:


# Pie chart for body style distribution

colors = ['gold', 'magenta', 'cyan', 'green', 'blue', 'red']  # Add more colors if you have more categories

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(car_body_style.values, labels=car_body_style.index, autopct='%1.1f%%', startangle=90, colors=colors)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title('Distribution of Car Body Style')

plt.show()


# In[14]:


# Countplot of categorical columns

fig, axis = plt.subplots(4,2,figsize=(15,15))

for i, ax in enumerate(axis.flatten()):
    if i < len(cat_columns):
        sns.countplot(data=car_df, x=cat_columns[i], ax=ax, palette='Set1')
        ax.set_title(f'Countplot of {cat_columns[i]}')

plt.tight_layout()
plt.show()


# ### Numerical EDA (Cars)

# In[15]:


# Checking for correlation between numerical columns
corr_matrix = car_df.corr()

# Create a heatmap using seaborn library
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# In[16]:


# Histplot of Numerical columns

num_columns = ['Annual Income', 'Price ($)']

fig, axis = plt.subplots(1,2,figsize=(8,4))

for i, ax in enumerate(axis.flatten()):
    if i < len(num_columns[:2]):
        sns.histplot(data=car_df, x=num_columns[:2][i], ax=ax, kde=True)
        ax.set_title(f'Histogram of {num_columns[i]}')
        
plt.tight_layout()
plt.show()


# ### Standardizing / Cleaning Data (Crude Oil, Unemployment & IR)

# In[17]:


# Function to clean covariates data

def extend_and_fill_dates(df, date_col, target_col):
    """
    Extend the DataFrame to cover a specific period and fill missing dates.
    
    Parameters:
    - df: The input DataFrame.
    - date_col: The name of the column containing dates.
    - target_col: The name of the target column to fill for missing dates.
    
    Returns:
    - A DataFrame extended to cover from January 2, 2022, to December 31, 2023, with missing dates filled.
    """
    # Ensure proper date type
    df[date_col] = pd.to_datetime(df[date_col])

    # Sorting Dates by ascending order (earlier to latest)
    df = df[[date_col, target_col]].sort_values(by=date_col)

    # Filter the DataFrame for dates from 2022 onwards, ending last day of 2023
    df = df[(df[date_col] >= pd.Timestamp('2022-01-01')) & 
            (df[date_col] <= pd.Timestamp('2023-12-31'))].reset_index(drop=True)
    
    # Define the start date
    start_date = pd.Timestamp('2022-01-02')
    
    # Check if the starting date is before 2022-01-02
    if df[date_col].min() > start_date:
        # Generate the missing dates from start_date to the day before the current first date in DataFrame
        missing_dates = pd.date_range(start=start_date, end=df[date_col].min() - pd.Timedelta(days=1), freq='D')

        # Create a DataFrame for the missing dates with NaN for the target column
        missing_dates_df = pd.DataFrame(missing_dates, columns=[date_col])
        missing_dates_df[target_col] = None  # Assign NaN to the target column

        # Concatenate the original DataFrame with the DataFrame of missing dates
        # Ensure to concatenate such that missing dates come first
        df = pd.concat([missing_dates_df, df], ignore_index=True)

        # Sort the DataFrame again by date to ensure correct order
        df.sort_values(by=date_col, inplace=True)

        # Backfill the target column to fill NaN values with the next valid value
        df[target_col].fillna(method='bfill', inplace=True)
    
    
    # Check if the last date is before 2023-12-31
    if df[date_col].max() < pd.Timestamp('2023-12-31'):
        # Generate the missing dates
        missing_dates = pd.date_range(start=df[date_col].max() + pd.Timedelta(days=1), end='2023-12-31')
        
        # Create a DataFrame for the missing dates with NaN for the target column
        missing_data = pd.DataFrame({date_col: missing_dates, target_col: [pd.NA] * len(missing_dates)})
        
        # Concatenate the original DataFrame with the missing data
        df = pd.concat([df, missing_data], ignore_index=True)
        
        # Sort by date if needed
        df.sort_values(by=date_col, inplace=True)

    # Set date_col as the index
    df.set_index(date_col, inplace=True)

    # Reindex the DataFrame to fill in missing dates
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(date_range)
    
    # Fill missing values in other columns with the previous day's info
    df.fillna(method='ffill', inplace=True)
    
    # Drop the specific date 2022-01-01 if present (to match car df) 
    df = df.drop(pd.Timestamp('2022-01-01'), errors='ignore')
    
    return df

def standardize_column_names(df):
    """
    Standardizes the column names of a DataFrame by making them lowercase,
    replacing spaces with underscores, removing content within brackets,
    and trimming leading/trailing spaces and underscores.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - The DataFrame with standardized column names.
    """
    df.columns = df.columns.str.lower()  # Convert column names to lowercase
    df.columns = df.columns.str.replace(' ', '_', regex=False)  # Replace spaces with underscores
    df.columns = df.columns.str.replace('\(.*?\)', '', regex=True)  # Remove brackets and contents within
    df.columns = df.columns.str.strip()  # Remove leading or trailing spaces
    df.columns = df.columns.str.rstrip('_')  # Remove any trailing underscores
    
    return df


# In[18]:


final_crude_df = extend_and_fill_dates(df=crude_df, 
                                       date_col='Date', 
                                       target_col='Crude_oil_Price')

final_unemploy_df = extend_and_fill_dates(df=unemploy_df, 
                                          date_col='DATE', 
                                          target_col='UNRATE')

# Using only 1 month yield data
final_yield_df = extend_and_fill_dates(df=yield_df, 
                                          date_col='Date', 
                                          target_col='1 Mo')


# ### Standardizing / Cleaning Data (Cars)

# In[19]:


# Standardizing column names
car_df_cleaned = standardize_column_names(df=car_df)

# Dropping of irrelevant columns due to lack of predictiveness, high cardinality
car_df_cleaned = car_df.drop(columns=['model', 'gender', 'car_id', 'customer_name', 'dealer_name', 'dealer_no', 'phone', 'dealer_region'], inplace=False)

# For mac m1
car_df['annual_income'] = car_df['annual_income'].astype(np.float32)
car_df['price'] = car_df['price'].astype(np.float32)

# Standardizing data (engine), double overhead camshaft as DOC and overhead camshaft as OC
car_df_cleaned['engine'] = car_df_cleaned['engine'].replace({'DoubleÂ\xa0Overhead Camshaft': 'DOC', 'Overhead Camshaft': 'OC'})


# In[19]:


car_df_cleaned


# ### Aggregating Cols

# In[20]:


# Calculate the mode for other categorical variables
aggregations = {
    'annual_income': 'mean',
    'price': 'mean',
    'company': lambda x: x.mode()[0] if not x.empty else x,
    'engine': lambda x: x.mode()[0] if not x.empty else x,
    'transmission': lambda x: x.mode()[0] if not x.empty else x,
    'color': lambda x: x.mode()[0] if not x.empty else x,
    'body_style': lambda x: x.mode()[0] if not x.empty else x,
}

# Applying the aggregation
final_df = car_df_cleaned.groupby('date').agg(aggregations).reset_index()

# Setting date to datetime format
final_df['date'] = pd.to_datetime(final_df['date'])

# Sorting values by date (ascending)
final_df = final_df.sort_values(by='date', ascending=True).reset_index()


# Converting 'date' column to datetime format and set it as the index
final_df['date'] = pd.to_datetime(final_df['date'])
final_df.set_index('date', inplace=True)

# Reindex the DataFrame to fill in missing dates
# Create a date range that covers the full period from min to max date
date_range = pd.date_range(start=final_df.index.min(), end=final_df.index.max(), freq='D')

# Reindex the DataFrame using the complete date range, filling missing values with NaNs
final_df = final_df.reindex(date_range)

# Fill missing values in other columns with the previous day's info
final_df.fillna(method='ffill', inplace=True)

# Dropping index col
final_df = final_df.drop(['index'], axis=1)


# ### Concat car df with Crude Oil, Unemployment & IR

# In[21]:


# Join the DataFrames on their index ('Date')
merged_df = final_df.join(final_crude_df, how='left')
merged_df = merged_df.join(final_unemploy_df, how='left')
merged_df = merged_df.join(final_yield_df, how='left')
merged_df = standardize_column_names(df=merged_df)


# In[22]:


merged_df


# ### Encoding

# In[23]:


# Splitting df for predictions and covariates
pred_df = merged_df[['annual_income', 'price']]
cov_df = merged_df[['company', 'color', 'engine', 'transmission', 'body_style', 'crude_oil_price', 'unrate', '1_mo']]


# Specifying of categorical columns for one hot encoding
categorical_columns = ['company', 'engine', 'transmission', 'color', 'body_style']

# Perform one-hot encoding on categorical cols for pred_df and cov_df
cov_df_encoded = pd.get_dummies(cov_df, columns=categorical_columns)
df_encoded = pd.get_dummies(merged_df, columns=categorical_columns)

# List of columns to exclude from being converted to categorical
exclude_cols = ['price', 'annual_income', 'crude_oil_price', 'unrate', '1_mo']

# Converting cols to categorical
for col in cov_df_encoded.columns:
    if col not in exclude_cols:
        cov_df_encoded[col] = cov_df_encoded[col].astype('category')
        
# Exclude the 'date' column from the list of columns
value_cols_pred = [col for col in pred_df.columns]
value_cols_cov = [col for col in cov_df_encoded.columns]
value_cols = [col for col in df_encoded.columns]


# ### Series Creation

# In[24]:


# Creating series for modelling
series_pred = TimeSeries.from_dataframe(df=pred_df, 
                                       time_col=None, 
                                       value_cols=value_cols_pred,
                                       fill_missing_dates=False, 
                                       freq='D')
  

series_cov = TimeSeries.from_dataframe(df=cov_df_encoded, 
                                       time_col=None, 
                                       value_cols=value_cols_cov,
                                       fill_missing_dates=False, 
                                       freq='D')


series = TimeSeries.from_dataframe(df=df_encoded, 
                                   time_col=None, 
                                   value_cols=value_cols,
                                   fill_missing_dates=False, 
                                   freq='D')


# ### Train Test Split and Covariates

# In[25]:


# Create training and validation sets
training_cutoff = pd.Timestamp("20231201")
train, val = series_pred.split_before(training_cutoff)
train = train.astype(np.float32)
val = val.astype(np.float32)


# In[26]:


# Normalize the time series
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series_pred)

# Changing type to float32 for standardization
series_transformed = series_transformed.astype(np.float32)


# In[27]:


# create year, month and integer index covariate series
covariates = datetime_attribute_timeseries(series_pred, attribute="year", one_hot=False)
covariates = covariates.stack(
    datetime_attribute_timeseries(series_pred, attribute="month", one_hot=False)
)
covariates = covariates.stack(
    TimeSeries.from_times_and_values(
        times=series_pred.time_index,
        values=np.arange(len(series)),
        columns=["linear_increase"],
    )
)

# Convert to float32 for standardization
covariates = covariates.astype(np.float32)
series_cov = series_cov.astype(np.float32)


# In[28]:


# transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
scaler_covs = Scaler()
scaler_covs_past = Scaler()
cov_train, cov_val = covariates.split_after(training_cutoff)
past_cov_train, past_cov_val = series_cov.split_after(training_cutoff)

scaler_covs.fit(cov_train)
scaler_covs_past.fit(past_cov_train)

covariates_transformed = scaler_covs.transform(covariates)
past_covariates_transformed = scaler_covs_past.transform(series_cov)


# ### Model Params and Training

# In[29]:


# default quantiles for QuantileRegression
quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]

input_chunk_length = 1
forecast_horizon = 30

num_samples = 10
figsize = (9, 6)


# ### Initial TFT Model without Covariates and EarlyStopping

# In[30]:


# Initial model
initial_model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=forecast_horizon,
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=16,
            n_epochs=100,
            add_relative_index=False,
            add_encoders=None,
            likelihood=QuantileRegression(
                quantiles=quantiles
            ),  # QuantileRegression is set per default
            random_state=42,
        #     pl_trainer_kwargs={"callbacks": [my_stopper]}
        )


# In[31]:


# Fitting Model
# initial_model.fit(train_transformed, 
#              # past_covariates=past_covariates_transformed,
#              future_covariates=covariates_transformed, 
#              verbose=True)


# In[31]:


# Saving/Loading initial model
# initial_model.save("TFT_Model_100P_No_Cov")
initial_model = TFTModel.load('TFT_Model_100P_No_Cov')


# In[40]:


# Backtesting predictions
backtest_series = initial_model.historical_forecasts(
    series_transformed,
    start=train.end_time() + train.freq,
    num_samples=num_samples,
    forecast_horizon=forecast_horizon,
    stride=forecast_horizon,
    last_points_only=False,
    retrain=False,
    verbose=True,
)


# In[41]:


# Backtesting function
def eval_backtest(backtest_series, actual_series, horizon, start, transformer, plot_time_start):
    
    # Inverse transform the entire multivariate series first
    actual_series_inversed = transformer.inverse_transform(actual_series)
    backtest_series_inversed = transformer.inverse_transform(backtest_series)

    # Set the start date for x-axis limitation
    start_date_for_plot = Timestamp(plot_time_start)
    
    # Iterate over components and plot separately (predicted variables, annual_income and price)
    for component in actual_series.components:
        plt.figure(figsize=figsize)
        actual_component_series = actual_series_inversed[component]
        backtest_component_series = backtest_series_inversed[component]

        # Plot the actual and backtest series for the current component
        actual_component_series.plot(label=f"Actual {component}")
        backtest_component_series.plot(label=f"Predicted {component}")
        
        # Customize and display the plot
        plt.title(f"Backtest {component} starting {start}, horizon: {horizon}")
        
        # Set x-axis limits
        plt.xlim(start_date_for_plot, actual_component_series.end_time())

        plt.legend()
        plt.show()

        # Calculate and print the MAPE for the current component
        mape_value = mape(actual_component_series, backtest_component_series)
        print(f"MAPE for {component}: {mape_value:.2f}%")

# Calling backtesting function
eval_backtest(
    backtest_series=concatenate(backtest_series),
    actual_series=series_transformed,
    horizon=forecast_horizon,
    start=training_cutoff,
    transformer=transformer,
    plot_time_start='2023-10-01'
)


# The actual data might be volatile, meaning it has large fluctuations that can be difficult to predict. This can make the forecasting graph look "busy" or "noisy," which might be interpreted as a "bad" graph even if the MAPE is relatively low.

# ### TFT model with Covariates

# In[42]:


my_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=16,
    n_epochs=100,
    add_relative_index=False,
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=quantiles
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    random_state=42,
#     pl_trainer_kwargs={"callbacks": [my_stopper]}
)


# In[72]:


# Fitting Model
# my_model.fit(train_transformed, 
#              past_covariates=past_covariates_transformed,
#              future_covariates=covariates_transformed, 
#              verbose=True)


# In[43]:


# Saving / Loading Model as required
# my_model.save("TFT_Model_100P")
my_model = TFTModel.load('TFT_Model_100P')


# In[51]:


# Backtesting predictions
backtest_series = my_model.historical_forecasts(
    series_transformed,
    past_covariates=past_covariates_transformed,
    future_covariates=covariates_transformed,
    start=train.end_time() + train.freq,
    num_samples=num_samples,
    forecast_horizon=forecast_horizon,
    stride=forecast_horizon,
    last_points_only=False,
    retrain=False,
    verbose=True,
)


# In[52]:


# Calling backtesting function
eval_backtest(
    backtest_series=concatenate(backtest_series),
    actual_series=series_transformed,
    horizon=forecast_horizon,
    start=training_cutoff,
    transformer=transformer,
    plot_time_start='2023-10-01'
)


# The actual data might be volatile, meaning it has large fluctuations that can be difficult to predict. This can make the forecasting graph look "busy" or "noisy," which might be interpreted as a "bad" graph even if the MAPE is relatively low.

# ### Final Model with Covariates and Early Stopping

# In[53]:


# stop training when validation loss does not decrease more than 0.001 (`min_delta`) over
# a period of 50 epochs (`patience`)
my_stopper = EarlyStopping(
    monitor="train_loss",
    patience=50,
    min_delta=0.001,
    mode='min',
)


# In[54]:


final_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=16,
    n_epochs=100,
    add_relative_index=False,
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=quantiles
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    random_state=42,
    pl_trainer_kwargs={"callbacks": [my_stopper]}
)


# In[59]:


# Fitting Model
# final_model.fit(train_transformed, 
#              past_covariates=past_covariates_transformed,
#              future_covariates=covariates_transformed, 
#              verbose=True)


# In[59]:


# Saving / Loading Model as required
# final_model.save("Final_TFT_Model")
final_model = TFTModel.load('Final_TFT_Model')


# In[70]:


# Backtesting predictions
backtest_series = final_model.historical_forecasts(
    series_transformed,
    past_covariates=past_covariates_transformed,
    future_covariates=covariates_transformed,
    start=train.end_time() + train.freq,
    num_samples=num_samples,
    forecast_horizon=forecast_horizon,
    stride=forecast_horizon,
    last_points_only=False,
    retrain=False,
    verbose=True,
)


# In[71]:


# Calling backtesting function
eval_backtest(
    backtest_series=concatenate(backtest_series),
    actual_series=series_transformed,
    horizon=forecast_horizon,
    start=training_cutoff,
    transformer=transformer,
    plot_time_start='2023-10-01'
)


# ### Base Model Sarima

# In[72]:


# Train test split for Arima in DF instead of series
training_cutoff = pd.Timestamp("20231201")
train_sarima = merged_df[:training_cutoff]
val_sarima = merged_df[training_cutoff:]   


# In[75]:


# Assuming 'merged_df' is your DataFrame and it has a DateTime index
# If not, you would need to convert the date column to a DateTime index:
# merged_df['date'] = pd.to_datetime(merged_df['date'])
# merged_df.set_index('date', inplace=True)

# Splitting the data into training and validation sets
train = merged_df.loc[:'2023-11-30']
validation = merged_df.loc['2023-12-01':]

# We assume the series is at least stationary or has been made stationary
# If not, consider differencing the series or transforming it (e.g., log transformation)
# Example of differencing:
# train['annual_income_diff'] = train['annual_income'].diff().dropna()

# Fit a SARIMA model (example parameters used, may require optimization)
# These parameters (p, d, q), (P, D, Q, s) should ideally be identified via model selection processes like AIC, BIC or cross-validation
sarima_model = SARIMAX(train['annual_income'], 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_result = sarima_model.fit(disp=False)

# Forecasting
forecast = sarima_result.get_forecast(steps=len(validation))
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Calculate MAPE
mape = np.mean(np.abs((validation['annual_income'] - forecast_values) / validation['annual_income'])) * 100

# Print MAPE
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

plt.figure(figsize=(10, 6))
plt.plot(train['annual_income'], label='Training Data')
plt.plot(validation['annual_income'], label='Actual Data')
plt.plot(forecast_values, label='Forecasted Data')
plt.fill_between(forecast_values.index,
                 forecast_conf_int.iloc[:, 0],
                 forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Annual Income (Actual vs Predicted)')
plt.legend()
plt.show()


# In[76]:


# Assuming 'merged_df' has a DateTime index and no missing values
# If the index is not a DateTime, convert it before proceeding

# Splitting the data into training and validation sets
train = merged_df.loc[:'2023-11-30']
validation = merged_df.loc['2023-12-01':]

# Fit a SARIMA model on the 'price' column
sarima_model_price = SARIMAX(train['price'], 
                             order=(1, 1, 1), 
                             seasonal_order=(1, 2, 1, 12),
                             enforce_stationarity=False,
                             enforce_invertibility=False)

sarima_result_price = sarima_model_price.fit(disp=False)

# Forecasting the price
forecast_price = sarima_result_price.get_forecast(steps=len(validation))
forecast_values_price = forecast_price.predicted_mean
forecast_conf_int_price = forecast_price.conf_int()

# Calculate MAPE for the 'price' column
mape_price = np.mean(np.abs((validation['price'] - forecast_values_price) / validation['price'])) * 100

# Print MAPE for the 'price' forecast
print(f'Mean Absolute Percentage Error (MAPE) for Price: {mape_price:.2f}%')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(train['price'], label='Training Data for Price')
plt.plot(validation['price'], label='Actual Price Data')
plt.plot(forecast_values_price, label='Forecasted Price Data')
plt.fill_between(forecast_values_price.index,
                 forecast_conf_int_price.iloc[:, 0],
                 forecast_conf_int_price.iloc[:, 1], color='lightgreen', alpha=0.5)
plt.title('Price (Actual vs Predicted)')
plt.legend()
plt.show()


# ### Auto Sarima Model

# In[73]:


# Building Sarima Model for annual income
annual_income_sarima = auto_arima(
    train_sarima['annual_income'], 
    start_p=1,  # Initial guess for the non-seasonal AR order
    start_q=1,  # Initial guess for the non-seasonal MA order
    max_p=3,    # Maximum non-seasonal AR order
    max_q=3,    # Maximum non-seasonal MA order
    m=12,       # Seasonal period (this is an example, adjust to your data)
    start_P=1,  # Initial guess for the seasonal AR order
    start_Q=1,  # Initial guess for the seasonal MA order
    max_P=3,    # Maximum seasonal AR order
    max_Q=3,    # Maximum seasonal MA order
    d=0,        # Non-seasonal differencing order (consider 1 if the data is not stationary)
    D=1,        # Seasonal differencing order (often 1 for annual data)
    seasonal=True,  # Indicate that we want to fit a seasonal ARIMA
    stepwise=True,  # Use stepwise algorithm for model selection
    trace=True,     # Print model fitting information
    error_action='warn',
    suppress_warnings=True,
    with_intercept=True  # Include an intercept in the model
)


# In[74]:


# Annual Income Arima Summary Stats
print(annual_income_sarima.summary())


# In[75]:


# Plotting the Diagnostics for Annual Income
annual_income_sarima.plot_diagnostics(figsize=(15,12))
plt.show()


# In[77]:


# Building Sarima Model for price
price_sarima = auto_arima(
    train_sarima['price'], 
    start_p=1,  # Initial guess for the non-seasonal AR order
    start_q=1,  # Initial guess for the non-seasonal MA order
    max_p=3,    # Maximum non-seasonal AR order
    max_q=3,    # Maximum non-seasonal MA order
    m=12,       # Seasonal period (this is an example, adjust to your data)
    start_P=1,  # Initial guess for the seasonal AR order
    start_Q=1,  # Initial guess for the seasonal MA order
    max_P=3,    # Maximum seasonal AR order
    max_Q=3,    # Maximum seasonal MA order
    d=0,        # Non-seasonal differencing order (consider 1 if the data is not stationary)
    D=1,        # Seasonal differencing order (often 1 for annual data)
    seasonal=True,  # Indicate that we want to fit a seasonal ARIMA
    stepwise=True,  # Use stepwise algorithm for model selection
    trace=True,     # Print model fitting information
    error_action='warn',
    suppress_warnings=True,
    with_intercept=True  # Include an intercept in the model
)


# In[78]:


# Plotting the Diagnostics for Price
price_sarima.plot_diagnostics(figsize=(15,12))
plt.show()


# In[79]:


# Price Arima Summary Stats
print(price_sarima.summary())


# From what I can see in the plot, the residuals appear to fluctuate around the zero line without any obvious patterns or systematic structure, which is a good sign. There aren't clear trends, seasonal effects, or periods of high volatility that would indicate model inadequacies. However, there are a few spikes that stand out, but these are relatively few and could be outliers or anomalies in the data.

# In[80]:


def forecast(SARIMA_model, periods, target_col, train_data, val_data, plot_start_date):
    """
    Forecast future values using an SARIMA model, calculate the MAPE for the forecast,
    and plot the training data and forecast starting from a specified date.
    
    Parameters:
    SARIMA_model: The trained SARIMA model.
    periods: The number of periods to forecast.
    target_col: The name of the target column to predict.
    train_data: The training dataset containing the target column.
    val_data: The validation dataset containing the target column.
    plot_start_date: The start date for plotting the forecast and training data.
    """
    # Forecast
    n_periods = periods
    fitted, confint = SARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(train_data.index[-1] + pd.DateOffset(days=1), periods=n_periods, freq='D')
    
    # Make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    # Plot training data from plot_start_date onwards
    plt.plot(train_data[target_col][plot_start_date:], color='#1f76b4')
    plt.plot(fitted_series, color='red')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

    plt.title(f'{target_col} - Forecast from {plot_start_date}')
    plt.show()

    # Generate predictions for the validation set
    predictions = SARIMA_model.predict(n_periods=len(val_data))
    
    # Ensure that the actual values and predictions have the same length
    actual_values = val_data[target_col].values
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(actual_values, predictions) * 100
    
    # Print the MAPE
    print(f"The MAPE of the SARIMA model for the {target_col} is: {mape:.2f}%")

forecast(annual_income_sarima, 30, 'annual_income', train_sarima, val_sarima, '2023-10-01')
forecast(price_sarima, 30, 'price', train_sarima, val_sarima, '2023-10-01')

