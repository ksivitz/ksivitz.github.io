## Predicting Sales With Time Series Forcasting  

### Project description:

The purpose of this project is to analyze sales data from 54 branches of Corporación Favorita, an Ecuadorian grocery store chain, and use this data to project sales. This project explores the concept of time series forecasting, as well as building hybrid models from multiple parallel time series. We will be using a Linear Regression model to predict the seasonality and trend of our time series, and XGBoost to predict the residual data from our initial model. 

### 1. The Data

This project consists of multiple related datasets. The main dataset includes sales data from 54 different stores across 33 categories, including automotive care, books, seafood, produce, and many more. Also provided are a few supplemental datasets, including local, regional, and national holiday information for Ecuador, oil prices for the given timeframe, and information about each individual branch of Corporatión Favorita. 

The first step in exploring our data is to look at the total sales across all stores and categories for the time period provided (1-1-2013 through 7-15-2017). As you can see from the following plot, sales follow a few different trends. Overall sales have steadily increased over time, with yearly and possibly weakly cycles evident as well. 


<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/total-sales-plot.PNG?raw=true"/>

Looking more closely at individual category sales, we can see that they follow a similar trend, however each category also has its own unique variables that affect the overall plot. Automotive, for instance, follows rather closely with total sales data, while baby care was not available till much later in our timeframe and has dropped to 0 multiple times before establishing itself as a stable category.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/total-sales-automotive.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/total-sales-babycare.PNG?raw=true"/>

Another plotting feature we can use to discover trends is a periodogram. A periodogram plots the variance across different time periods throughout the year, including annually, monthly, weekly, etc. The more variance our series has at each of these periods, the more relevant they are.

Looking at a periodogram for our total sales, we can see that the largest variance is in weekly and biweekly intervals, with annual and quarterly trends present as well. In produce however, you can see that the variance is largest annual, dropping off around the bi-weekly (6) mark, and picking back up in the weekly / bi-weekly intervals. 


<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/total-sales-periodogram.PNG?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/produce-periodogram.PNG?raw=true"/>

Another variable that we can use to predict trends is called a lag feature (sales data from the previous day). Including sales data from the previous day or multiple days can show which direction the sales trend is moving and by what interval it is increasing / decreasing. To determine how many lag variables are optimal for our series, we can use a partial autocorrelation plot to determine how correlated each lag is to our current days sales. From the following plot for total sales, we can see that data from up to 9 days previous contains useful information in determining our sales.


<img src="https://github.com/ksivitz/ksivitz.github.io/blob/4465f632e5ccece393ede848349e4b77431b12a1/images/sales/partial-cor-total-sales.PNG?raw=true"/>


After computing the optimal periods and lag features for each store/category combination, we can see from the following histograms that a majority of our categories have a optimal seasonality of 12 (monthly), and an average optimal number of lags of 8, with a majority skewing right (higher). Rather than use separate values for each combination, we will set our prediction values of 12 for periods and 9 for autocorrelation. 


<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/period-histogram.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/pcf-value-histogram.PNG?raw=true"/>

## Prediction Part 1 - Seasonality and Trends


We are now ready to create our first model. We will be using Linear Regression to compute the seasonality and trend of our sales data for each individual store / family combination and then calculate the sum of these predicted series to get our total sales data. The following plot shows our predictions compared to the actual sales provided. We have a RMSE of 101,110.83 for our training data, with our test data doing slightly better at 87,106.68. Our testing period does not include the holiday season, which is presumably why our testing RMSE is significantly better than training, as the holidays is where the largest amount of variance occurs. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/total-sales-seasonal-forcast.PNG?raw=true"/>

When we subtract our predicted sales values from the actual sales, we are left with our residual (leftover) data that we were unable to accurately predict with seasonality and trends. This data is what we will attempt to predict with the second part of our hybrid model. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/total-sales-residual-plot.PNG?raw=true"/>

## Prediction Part 2: Preparing Residual Data

For our second model we will be using XGBoost, a series of boosted decision trees, to make predictions on the remaining sales not picked up by our linear regression model. Before we do this however, we will be adding the following supplemental data to our dataset:

Promotional Information<br>
Holidays (local, regional, and national)<br>
New Years Day<br>
Day of the Week<br>
Month of the Year<br>
Monthly Max Temperature<br>
Monthly Precipitation<br>
Current Price of Oil<br>
Payday (yes or no)<br>
Earthquake (a large earthquake struck Ecuador on 4/16/2016, effecting store sales for several weeks afterwards)<br>
Store Number<br>
Family (Product Category)<br>

Additionally, we will create lag features to help predict any trends found in our residuals.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/2cd3b143328ff8dc274eb271458a265c80e73ee3/images/sales/x-test.PNG?raw=true"/>

## Prediction Part 3: Predicting Residual Data

For our linear regression model, we created a separate model for each store / family combination and predicted them individually. For XGBoost, this would require much more computing power, so I have decided to train one model to be used on all of our time series. Because the amount of residual sales can vary widely between different categories, we will standardize our residual data and predict the standard deviation, rather than the actual residual amount. 

Once our data is reformatted and split into our train, test, and validation sets, we are ready to begin training our model. We will be using Amazon Sagemaker to do our training and host our trained model for predictions. I have run a hyperparameter tuning job to determine the optimal hyperparameters to use for our model. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/hyperparameters.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/model-training.PNG?raw=true"/>

## Evaluation on Test Set

Now that we have trained our model and predicted our residuals, it is time to evaluate the model’s performance. 

The first step in evaluation is to see how well the model performed on our test data. Looking at the following graph, you can see that our predictions match well with the residual data. The RMSE for our standardized residuals is .75569, or roughly .75 standard deviations. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/2cd3b143328ff8dc274eb271458a265c80e73ee3/images/sales/resid-vs-pred.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/b0fed4b9fcd784ad2d1522ebde19f04e10d02c04/images/sales/residual-rmse.PNG?raw=true"/>

When we combine these residual predictions with our seasonality predictions, we can see that the hybrid model does a much better job of predicting the sales data than our linear regression model alone, with the root mean squared error of the hybrid model being almost half of that of the seasonal-only model.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/resid-plus-sales.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/35e0b03c7766f3005e2ea24740ca5bdd523bc0f4/images/sales/rmse-predicted.PNG?raw=true"/>

## Evaluation on Validation Set

The final step in evaluating our hybrid model is to view its performance on our validation (holdout) dataset. This dataset contains feature data for 15 days, from 7-16-2017 through 7-30-2017.

For our previous evaluation, we used lag variables provided by our test data. However, because we do not have actual lag values for any prediction dates beyond the first in our validation set, we will need to make predictions for the first date in our set (7-16-2017) and then use those predicted values as lags for the next date's predictions, and so on. The following plot shows our predictions vs actual residual values for our validation date range. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/12ad6def5a2ded8544c26ba25cfb12fe76a2068c/images/sales/residuals-vs-predicted-val.PNG?raw=true"/>

Because we used our predictions as lag features, there is a risk of compounding errors as we move further into our validation dataset. The following plot shows how our predictions from our projected lags compare to predictions with the actual residual lag variables. As you can see, our projected lag predictions begin to diverge after 9 days, however the RMSE for our projected vs actual lag predictions is insignificant (less than 1%).

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/12ad6def5a2ded8544c26ba25cfb12fe76a2068c/images/sales/resiudal-predictions-lags.PNG?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/12ad6def5a2ded8544c26ba25cfb12fe76a2068c/images/sales/rmse-lags.PNG?raw=true"/>

Finally, we will make our seasonality sales predictions and combine them with our residual sales predictions. Again, because we use lag variables in our prediction input, we will need to either predict daily and add our predictions to our lag variables, or use multi-step prediction, which will predict values for however many steps in the future specified. For this model, we will be using multi-step prediction. By making these predictions and adding them to our residual predictions, we can achieve a RMSE of only $45,283 per day, or an error of roughly 7% of total sales.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/12ad6def5a2ded8544c26ba25cfb12fe76a2068c/images/sales/final-sales-predictions.PNG?raw=true"/>

Below is the notebook containing the full workup of this project

[Time Series Forecasting](https://ksivitz.github.io/notebooks/Forecasting-Supermarket-Sales.html).
