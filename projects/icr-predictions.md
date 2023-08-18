## ICR - Identifying Age Related Conditions

### Project description:

The purpose of this project is to determine if a patient has one or more of three age-related conditions based on 50 anonymized data points. 

### 1. The Data

For this project I will be using a dataset provided by Kaggle, an online coding competition website. Because the data is medical and contains personal information, all data is encoded and anonymized. For this reason all of our data exploration and feature engineering will be based solely on the numbers, and not involve any domain knowlege. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/2a346a620828f8646795d597f838a3c69e5cfcb7/images/icr-data.PNG?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/2a346a620828f8646795d597f838a3c69e5cfcb7/images/icr-data-2.PNG?raw=true"/>

The first step in preparing this data looking for missing values. As we can see from the following chart, there are a few categories with one or two missing values, and one category (EL) with 60 missing values. because we have no knoweldge of what each category represents, I decided the best way to fill these missing values is with a KNN imputer. This program fills missing values with the value of their nearest neighbor in terms of all other categories. 


Now that we have computed our missing values, we can explore our data. The first step is to view the distribution of our data categories. Below are a few histograms of some randomly selected categories. As you can see from these plots, the data is fairly normally distributed, with outliers tending to be rightly skewed. This suggests standardization will be the best method for scaling our data when preparing it for our model. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/3db595d655e66b1107688a878849afe7f9a05bf7/images/dl-hist.PNG?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/3db595d655e66b1107688a878849afe7f9a05bf7/images/ee-hist.PNG?raw=true"/>
<img src="https://github.com/ksivitz/ksivitz.github.io/blob/3db595d655e66b1107688a878849afe7f9a05bf7/images/cr-hist.PNG?raw=true"/>

Next we will look at feature correlation. I have taken our data and seperated it into 3 randomly selected groups of 10. With these subsets we can create some heatmaps that show how correlated our categories are to each other. As you can see from these plots, there is a fair amount of correlation, with some categories reaching the 70%-80% level. As for correlation to our target category, we have low but non-null correlation on some categories and up to 25% on others, suggesting a model should do fairly well at predicting our target class. 

<center><img src="https://github.com/ksivitz/ksivitz.github.io/blob/3db595d655e66b1107688a878849afe7f9a05bf7/images/heatmaps.PNG?raw=true"/></center>
  
### 2. Model Training and Evaluation

Now that the data has been organized and collected, it is time to train our model. Because we have a high number of categories with varying levels of correlation, I have decided to use XGBoost as our model, a series of boosted decission trees. Because our data is fairly normally distributed, I have scaled all of our categories using a standardization method, and converted our categorical column into dummy variables. Once the data is prepared, we can start to train our model. 

The first step is choosing hyperparameters for our model. I started by using sagemakers hyperparameter tuning to create a hyperparameter tuning job. The results gave us a model that performed very well, however it appeard to overfit to our training data. Because of this, I adjusted some of the hyperparameters to help reduce this overfitting. I increased the values of lambda and alpha (our regularization hyperparameters), and decressed the max depth from 50 to 45.

Using area under the curve (AUC) as our training metric, we can see that our model performs fairly well, with 99.55% on our training data and 95.93% on test data. From the confusion matrix we can see our model performed very well, only miss-classifying 2 of the 23 positive instances. 

<center><img src="https://github.com/ksivitz/ksivitz.github.io/blob/4a99524cf92a1aef693c89d25198b9383202f736/images/auc-curve-test.PNG?raw=true"/></center>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/4a99524cf92a1aef693c89d25198b9383202f736/images/confusion-test.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/4a99524cf92a1aef693c89d25198b9383202f736/images/class-report-test.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/4a99524cf92a1aef693c89d25198b9383202f736/images/loss-test.PNG?raw=true"/>

Finally, we will evaluate our model on the holdout (validation) set. As you can see there is a slight dropoff in performance, though our model still has an accruacy score of 89.5% and an AUC of 89%. From the confusion matrix we can see that we were able to correctly identify 87% (20 out of 23) positive instances.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/979f251ddfe52e03b3ceeb089f6e44d581f66dd2/images/auc-5.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/979f251ddfe52e03b3ceeb089f6e44d581f66dd2/images/confusion-5.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/979f251ddfe52e03b3ceeb089f6e44d581f66dd2/images/class-5.PNG?raw=true"/>

Because the purpose of this project is to correctly identify positive instances of age related conditions in patients, our most important metric is the recall value for our 
positive (1) class. XGBoost provides probabilities as its output, with values less that 0.50 resulting in a 0 classification and outcomes greater than 0.50 resulting in a 1 classification. By adjusting this threshold from 0.50 to 0.40, we can increase our likelyhood of identifing patients with a positive result from our test.  


<img src="https://github.com/ksivitz/ksivitz.github.io/blob/979f251ddfe52e03b3ceeb089f6e44d581f66dd2/images/auc-4.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/979f251ddfe52e03b3ceeb089f6e44d581f66dd2/images/confusion-4.PNG?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/979f251ddfe52e03b3ceeb089f6e44d581f66dd2/images/class-report-4.PNG?raw=true"/>


As you can see, lowering this threshold allowed us to increase our percentage of correctly identified positive instances from 87% to 91%.



In conclusion, modeling can help identify patients who have one of these age related conditions and provide recommendations for further screening. 

Below is the notebook containing the full workup of this project

[Reddit Sentiment vs Apple Price Notebook](https://ksivitz.github.io/notebooks/ICR-XGBoost-Notebook.html)
