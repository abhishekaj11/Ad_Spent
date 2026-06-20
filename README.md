# Ad Spent vs Sales Prediction (Linear Regression)

This is a small machine learning project I built while learning Python and scikit-learn. The idea is simple — given how much a company spends on TV, Radio and Newspaper ads, can we predict how many units of Sales they will get?

I am currently learning Data Analytics and ML on my own, and this is one of my practice/portfolio projects.

## About the Dataset

The dataset (`data/advertising.csv`) has 200 rows with these columns:

- `TV_Spent` — money spent on TV ads (in ₹ thousand)
- `Radio_Spend` — money spent on Radio ads (in ₹ thousand)
- `Newspaper_Spend` — money spent on Newspaper ads (in ₹ thousand)
- `Sales` — units sold

Note: I generated this data myself using a Python script (`generate_data.py`) with some randomness added, instead of downloading from Kaggle. I did this so I can clearly understand how the data is created and check if my model is actually learning the right relationship or not. The real classic version of this dataset is the "Advertising Dataset" used in many ML tutorials, so I followed the same structure.

## What I did

1. Loaded the data and checked basic stats (`.describe()`, `.head()`)
2. Checked correlation of each ad channel with Sales
3. Made some plots (heatmap, scatter plots) to understand the data before building any model
4. Split data into train (80%) and test (20%) sets
5. Built a Simple Linear Regression model using only TV spend
6. Built a Multiple Linear Regression model using all 3 ad channels
7. Compared both models using MAE, RMSE and R² score
8. Plotted Actual vs Predicted values and residuals to check if the model is doing well
9. Saved the trained model using `joblib`

## Files in this repo

```
ad_spend_regression/
│
├── data/
│   └── advertising.csv          # dataset
│
├── outputs/
│   ├── 01_correlation_heatmap.png
│   ├── 02_feature_scatter.png
│   ├── 03_simple_regression_line.png
│   ├── 04_actual_vs_predicted.png
│   ├── 05_residual_plot.png
│   └── sales_regression_model.pkl   # saved trained model
│
├── generate_data.py             # creates the dataset
├── regression_model.py          # main ML pipeline (this is the main file)
└── README.md
```

## How to run this project

1. Clone this repo
2. Install the required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Run the files in this order:

```
python generate_data.py
python regression_model.py
```

This will create the dataset, train both models, print out the results in terminal, and save all the plots + trained model inside the `outputs/` folder.

## Results I got

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Simple Linear Regression (TV only) | 2.33 | 2.83 | 0.675 |
| Multiple Linear Regression (TV + Radio + Newspaper) | 1.41 | 1.66 | 0.888 |

The multiple regression model performed much better than using TV spend alone. R² of 0.888 means the model is able to explain about 88.8% of the variation in Sales, which I think is a good result for this kind of project.

Final equation the model learned:

```
Sales = 6.83 + 0.044 * TV_Spend + 0.182 * Radio_Spend + 0.010 * Newspaper_Spend
```

## What I learned from this project

- How to do basic EDA before jumping into modeling
- Difference between simple and multiple linear regression
- Why we should never test a model on the same data it was trained on
- How to read R², MAE and RMSE and what they actually mean
- That not every feature is useful — Newspaper spend had almost no effect on Sales, which I confirmed both from correlation value and from the model's coefficient
- How to save a trained model using joblib so it can be reused later without retraining

## Future improvements

- Try this on a real dataset from Kaggle instead of generated data
- Try other models like Random Forest or XGBoost and compare results
- Add cross validation instead of single train/test split
- Build a simple web app (using Streamlit) where user can enter ad budget and get predicted sales

## About me

I am a fresher learning Data Analytics and Machine Learning. This project is part of my self learning journey and portfolio building. Feel free to check the code and suggest improvements.
