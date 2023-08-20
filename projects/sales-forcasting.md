## Predicting Sales With Time Series Forcasting  

### Project description:

The purpose of this project is to analyze store sales data from 54 stores across the country of Equador, and create a prediction model to predict future sales predictions of the predicted values. predict.  

### 1. The Data

The dataset provided by the company contains several useful columns with information about their customers. These columns are Age, Gender, Marital Status, Occupation, City Category, Time in City, Products Purchased, and Purchase Price. Using this data, we can begin to determine what kind of correlation we have between the various features and their relationship to the customers’ purchases.

The first column we will look at is purchase price. The following distribution plot shows that purchase amounts range from 0 to $25000, with the average spent skewing towards $5,000 - $10,000.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_dist.jpg?raw=true"/>

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/dist_prod.jpg?raw=true"/>




The next feature we will look is customer age. As you can see from these first two graphs, the majority of our purchases come from customers between the ages of 18-45, with age range 26-35 being the largest customer base. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_age.jpg?raw=true"/>


<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/top5age.jpg?raw=true"/>

However, each customer tends to spend roughly the same amount of money, regardless of their age group.  

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_avg_age.jpg?raw=true"/>

Another factor we can explore is occupation. The data has been split into 20 different job types, and it is clear from the following graph that what type of job a person has greatly affects the purchasing amount spent on Black Friday.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/purch_occ.jpg?raw=true"/>

We can also see from the following pie chart that the largest share of purchases come from customers living in city type B, with only a quarter of purchases coming from customers in city type A.

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/Pie_city.jpg?raw=true"/>

Finally, we will look at purchase amount by product. There are 3 distinct product categories, each with a range of individual products. As you can see from the following graphs, purchasing amount varies greatly by both product category and individual products. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/prod_purch.jpg?raw=true"/>

### 2. Model Training and Evaluation 

Now that we have explored our data, it is time to start predicting. Because we have a mix of both numerical and classification features, I have decided that a Random Forest Regression model is the best model to use in predicting customer purchasing habits. Using this model, we can get a prediction score of 68%, with a root mean squared error of $2,828. 

<img src="https://github.com/ksivitz/ksivitz.github.io/blob/ebc75764e30570dd709c10f43f48623710aaac96/images/rand_test_friday.JPG?raw=True"/>

Using this information, the company can determine which customers are most likely to make large black friday purchases, and create personalized promotions to draw these customers to their stores. 


Below is the notebook containing the full workup of this project

[Black Friday Purchase Predictions](https://ksivitz.github.io/notebooks/black_friday_notebook.html).
