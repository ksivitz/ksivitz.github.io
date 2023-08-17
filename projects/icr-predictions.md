## ICR - Identifying Age Related Conditions

### Project description:

The purpose of this project is to determine if a patient has one or more of three age-related conditions based on 50 anonymized data points. 

### 1. The Data

For this project I will be using a dataset provided by Kaggle, an online coding competition website. Because the data is medical and contains personal information, all data is encoded and anonymized. For this reason all of our data exploration and feature engineering will be based solely on the numbers, and not involve any domain knowlege. 

The first step in preparing this data looking for missing values. As we can see from the following chart, there are a few categories with one or two missing values, and one category (EL) with 60 missing values. because we have no knoweldge of what each category represents, I decided the best way to fill these missing values is with a KNN imputer. This program fills missing values with the value of their nearest neighbor in terms of all other categories. 

<img src=filled values something>

Now that we have computed our missing values, we can explore our data. The first step is to view the distribution of our data categories. Below are a few histograms of some randomly selected categories. As you can see from these plots, the data is fairly normally distributed, with outliers tending to be rightly skewed. This suggests standardization will be the best method for scaling our data when preparing it for our model. 

<img src=histogram plots>

Next we will look at feature correlation. I have taken our data and seperated it into 3 randomly selected groups of 10. With these subsets we can create some heatmaps that show how correlated our categories are to each other. As you can see from these plots, there is a fair amount of correlation, with some categories reaching the 70%-80% level. As for correlation to our target category, we have low but non-null correlation on some categories and up to 25% on others, suggesting a model should do fairly well at predicting our target class. 

<img src=heatmaps>

### 2. Model Training and Evaluation

Now that the data has been organized and collected, it is time to train our model. Because we have a high number of categories with varying levels of correlation, I have decided to use XGBoost as our model, a series of boosted decission trees. Because our data is fairly normally distributed, I have scaled all of our categories using a standardization method, and converted our categorical column into dummy variables. Once the data is prepared, we can start to train our model. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/error_rate.png?raw=true"/>

Using area under the curve (AUC) as our training metric, we can see that our model performs fairly well, with 99% on our training data and 93% on test data. 






In conclusion, although reddit sentiment does have a slightly better than 50% prediction rate on the movement of APPLE stock, it may be best to look for a different source for stock market trading advice. 

Below is the notebook containing the full workup of this project

[Reddit Sentiment vs Apple Price Notebook](https://ksivitz.github.io/notebooks/ICR-XGBoost-Notebook.html)
