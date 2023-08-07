## ICR - Identifying Age Related Conditions

### Project description:

The purpose of this project is to determine if a patient has one of 3 age related conditions based on 50 anonymized data points. 

### 1. The Data

For this project I will be using a dataset collected by SocialGrep containing comments with mentions of APPL from 10/31/2016 - 10/31/2021. Each row in this dataset contains a UTC Timestamp, the comment text, the subreddit the comment came from, and various other defining features. Our main focus from this dataset will be the timestamp and the comment text. 

The first step in preparing this data is to create a sentiment score for each comment. Once each comment was tokenized and lemmatized, I used SIA to create a polarity score for each word and combined these scores to receive a final sentiment score for each comment. Once the scores were tallied, I grouped the dataset by date and then combined the scores to create total sentiment for each trading day. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/sent.JPG?raw=true"/>

Looking at the combined scores vs the top 5 subreddits these comments were scraped from, we can see certain communities are much more optimistic than others, with the options trading subreddit WallStreetBets having the highest combined sentiment score. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/top5sent.jpg?raw=true"/>

Now that we have our sentiment scores, the next step is to collect price data for APPL during our comment period. This data was downloaded from Yahoo Finance and contains open, close, high, low, and volume for each day in our time period. After uploading the data to our workbook, I merged this data with our sentiment data and calculated the difference between today' and tomorrow’s closing price. Once this difference was calculated, I created a Class label for each date, where 1 represents a raise in stock price and 0 represents a lowering of the stock price. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/merged.JPG?raw=true"/>

As we can see from the following scatterplot, there is a clear correlation between the opening price difference and class, while the sentiment score looks fairly even across each value when compared to the class label. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/sent_open.jpg?raw=True"/>


### 2. Model Training and Evaluation

Now that the data has been organized and collected, it is time to see what correlation there is between these sentiment scores and the stock price. For the first model we will be using only the sentiment score vs the change in closing price. I chose K Nearest Neighbors for this model and collected the test error rate for k's 1-20 to determine the best number of neighbors to use for this data. As you can see, 5 appears to be the k value with the lowest error rate, so I set k=5 and tested the model against our data. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/error_rate.png?raw=true"/>

Using only the sentiment score, we get an overall accuracy of 54%. For trading, the most useful score would be the recall score of our 1 label, since the goal is to buy when there is a predicted rise in the stock’s price. Here we have a score of 55%.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/k_sent_only.JPG?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/plot_sent.JPG?raw=true"/>

The next step is be to combine these sentiment scores with other features available in our dataset to see if we can come up with a higher recall rate. To do this I added in the daily volume, as well as the opening price of the stock for the next trading day. Using K Nearest Neighbors with a k value of 30, I was able to raise the recall score for our 1 prediction to 65%. This is a much more useful score; however it requires you to wait until opening of the next day to place any trades, limiting potential profits. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/error_rate_all.jpg?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/class_report_all.JPG?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/plot_all_appl.JPG?raw=true"/>

Finally, I tested a Random Forest model to see if there was an increase in accuracy.  By doing a grid search for the best parameters for our data, I ran a Random Forest algorithm with a max depth of 6 and set the number of estimators to 10. This led to a recall score of 63% for our 1 label, slightly worse than that of K Nearest Neighbors.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/class_report_forest_friday.JPG?raw=true"/>

In conclusion, although reddit sentiment does have a slightly better than 50% prediction rate on the movement of APPLE stock, it may be best to look for a different source for stock market trading advice. 

Below is the notebook containing the full workup of this project

[Reddit Sentiment vs Apple Price Notebook](https://ksivitz.github.io/notebooks/ICR-XGBoost-Notebook.html)
