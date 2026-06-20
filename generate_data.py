import numpy as np
import pandas as pd

np.random.seed(48)

n=200

tv_spent = np.random.uniform(5,300,n)
radio_spent = np.random.uniform(0,50,n)
newspaper_spent = np.random.uniform(0,100,n)

noise = np.random.normal(0,2,n)

sales =(
    7
    +0.045 *tv_spent
    +0.19 *radio_spent
    +0.002 *newspaper_spent
    +noise
)

df = pd.DataFrame({
    "Tv_Spent" : tv_spent.round(2),
    "Radio_Spent" : radio_spent.round(2),
    "Newspaper_Spent" : newspaper_spent.round(2),
    "Sales" : sales.round(2),
})

df.to_csv(r"D:\AI\Ad_Spent/data/advertising.csv", index=False)
print('Saved dataset: ', df.shape)
print(df.head())
