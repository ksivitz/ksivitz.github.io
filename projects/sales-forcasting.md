## Predicting Sales With Time Series Forcasting  

### Project description:

The purpose of this project is to analyze sales data from 54 branches of Corporación Favorita, an Equidorian grocery store chain, and use this data to project sales. This project explores the concept of time series forcasting, as well as buiding hybrid models from multiple parellel time series. We will be using a Linear Regression model to predict seasonallity and trend of our time series, and XGBoost to predict the residual data from our initial model. 

### 1. The Data

This project consists of multiple related datasets. The main dataset includes sales data from 54 different stores across 33 categories, ranging from  automotive care to books, seafood, produce, and many more. Also included are a few suplimental datasets, including local, regional, and national holiday information for Equador, oil prices for the given timeframe, and information about each individual branch of Corporatión Favorita. 

The first step in exploring our data is to look at the total sales across all stores and categories for the time period provided (1-1-2013 through 7-15-2017). As you can see from the following plot, sales follow a few different trends. Overall sales have steadily increased over time, with yearly and possibly weakly cycles evident as well. 

total sales image
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_dist.jpg?raw=true"/>

Looking more closely at individal category sales, we can see that they follow a similar trend, however each category also has its own unique trends that it follows. Automotive, for instance, follows rather closely with total sales data, while baby care wasnt available till much later in our timeframe and has dropped to 0 multiple times before establishing itself as a stable category.

Automotive sales plot
baby care sales plot

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/dist_prod.jpg?raw=true"/>

Seasonallity?????

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_age.jpg?raw=true"/>

Another useful plotting feature we can use to discover trends is a periodogram. A periodogram plots the variance across different time periods throughout the year, including annually, monthly, weekly, etc. The more variance our series has at each of these periods, the more relevant they are.

Looking at a periodogram for our total sales, we can see that there is a strong variance in weekly and biweekly intervals, as well as anually and quarterly. In----- however, you can see that the variance is largest anually with ..... ( pick one other than baby care

periodogram total sales
periodogram other. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/top5age.jpg?raw=true"/>

The next thing? that we can use to predict trends are lag features. A lag is simplty the sales data from the previous day, so using yesterdays sales data to predict sales for today. Sometimes it is helpful to use multiple lags, or sales data from the previous 7-12 days, to determine a trend and help better predict todays sales. To determine how many lag variables are optimal for our series, we can use a pcf chart? to determine how correlated each lag is to our current days sales. From the following pdf lag grapg thing, we can see that data from up to 9 days previous contains useful information in determining our sales.

Lag graph thing

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_avg_age.jpg?raw=true"/>

After computing the optimal periods and lag features for each store/category combination, we can see from the following charts that a majority of our categories have a optimal seasonality of 12 (monthly), and an average optimal number of lags of 8, skewing right (higher). For this reason we will set our prediction values of 12 for periods and 9 for lags. 

period histogram
lag histogram


## prediction part 1 - seasonality and trends


We are now ready to compute our first model. We will be using Linear Regression to compute the seasonallity and trend of our sales data for each individual store / family combination and then compute the sum of these predicted series to get our total sales data. The following plot shows our predictions compared to acutal sales data. We have a RMSE of 101,110.83 for our training data, with our test data doing slightly better at 87,106.68. Our testing period does not include the holiday season, which is presumably why our testing RMSE is significatly better than training, as the holidays is where a large amount of variance occurs. 

total sales prediction plot
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_occ.jpg?raw=true"/>

When we subtract our predicted sales values from the actual sales, we are left with our residual (leftover) data that we were unable to accuratly predict with seasonallity and trends. This data is what we will attempt to predict with the second part of our hybrid model. 

residual data plot

## Prediction Part 2 : residuals

For our second model we will be using XGBoost, a series of boosted decission trees, to make predictions on the remaining sales not picked up by our first model. Before we do this however, we will be adding the following sublimental data to our dataset:

Promotional Information
Holidays (local, regional, and national)
New Years Day
Day of the Week
Month of the Year
Monthly Max Temperature
Monthly Percipitation
Current Price of Oil
Payday
Earthquake (an earthquake struck Equador on 4/16/2016, effecting store sales for serveral weeks
Store Number
Family (Product Category)

We will also be computing lag features for our data to help predict any trends found in our residuals. 

## Prediction

For our linear regression model, we created a seperate model for each store / family combination and predicted them individually. For XGBoost, this would require alot more computing power, so I have decided to train one model to be used on all of our time series. Because the amount of residual sales can vary widely between different categories, we will standardize our residual data and predict the standard deviation rather than the actual residual amount. 

dataframe head with standardized resid in front

Now that our data is reformated and split into our train, test, and validation sets, we are ready to begin training our model. We will be using Amazon Sagemaker to do our training and host our trained model for predictions. I have run a hyperparameter tuning job to determine the optimal hyperparameters to use for our model. 

hyperparameter tuning job results

model training results

## Evaluation

Now that we have trained our model and predicted our residuals, it is time to evaluate the models performance. 

The first step in evaluation is looking at how well the model performed on our test data. Looking at the following graph, you can see that our predictions match fairly well with the residual data. our RMSE for our standardized residuals is .75569, or about .75 standard deviations. 


Below is the notebook containing the full workup of this project

[Black Friday Purchase Predictions](https://ksivitz.github.io/notebooks/black_friday_notebook.html).