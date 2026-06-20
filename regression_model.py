import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

OUT = r"D:\AI\AD_SPENT\outputs"


df =pd.read_csv (r"D:\AI\AD_SPENT\data\advertising.csv")

print("=" * 54)
print("step one data review")
print("=" *60)
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns\n")

print(df.head())
print("\nSummary statistics: ")
print(df.describe().round(2))

# EDA

print("\n"+"=" *50)
print("check for sales connections")
print("="*60)

corr =df.corr()["Sales"].drop("Sales").sort_values(ascending=False)
print(corr.round(3))

# heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f', square=True)

plt.title("correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/01_correlation_heatmap.png", dpi=120)
plt.close()

fig, axes = plt.subplots (1, 3, figsize=(15,4))

for ax , col in zip (axes, ["Tv_Spent", "Radio_Spent", "Newspaper_Spent"]): 
    
    
    ax.scatter(df[col], df["Sales"], alpha=0.6, color="#378ADD")
    ax.set_xlabel(f"{col} ($ thousand)")
    ax.set_ylabel("Sales")
    ax.set_title(f"{col} vs Sales")
                  
plt.tight_layout()
plt.savefig(f"{OUT}/02_feature_scatter.png", dpi=120)
plt.close


 # Train/Test Split

x_simple = df[["Tv_Spent"]]
x_multi= df[["Tv_Spent", "Radio_Spent", "Newspaper_Spent"]] 
y =df["Sales"]

X_train_s, X_test_s, y_train, y_test = train_test_split(x_simple, y , test_size=0.2, random_state=42)
X_train_m, X_test_m, _,_ = train_test_split(x_multi,y, test_size=0.2, random_state=42)

print("\n" + "="*45)
print("train test")
print("-"*60)

print(f"Training rows: {len(X_train_s)} Test rows:{len(X_test_s)}")

#linear regresion

print("\n" "linear regression")
print("="*40)

model_simple=LinearRegression()
model_simple.fit(X_train_s, y_train)

m = model_simple.coef_[0]
b = model_simple.intercept_

print(f"Learned Equation: Sales = {m:.4f} *Tv_Spent + {b:.4f}")
print(f'In every Interpretaions : every extra $1000 spenton tv spend predicts the' f"{m:.3f} more units of sales, holding everything equal")

pred_simple = model_simple.predict(X_test_s)
mae_s = mean_absolute_error(y_test, pred_simple)
rmse_s = np.sqrt(mean_squared_error(y_test, pred_simple))
r2_s = r2_score(y_test, pred_simple)

print(f"\n Test set performance :")
print(f"MAE {mae_s:.3f} (average prediction of many sales)")
print(f"RMSE {rmse_s:.4f}")
print(f"R2 {r2_s:.3f} {r2_s*100:.1f}% sales varation expalained")

plt.figure (figsize=(7,5))
plt.scatter(X_train_s, y_train, alpha =0.3 , color="#378ADD", label= 'Training data' )
plt.scatter(X_test_s, y_test , alpha=0.5 , color="#1D9E75", label=" test data (held out) ")

x_range = np.linspace (df["Tv_Spent"].min (), df["Tv_Spent"].max(), 100). reshape(-1,1)
plt.plot(x_range, model_simple.predict(pd.DataFrame(x_range, columns =["Tv_Spent"])), color="#D85A30", linewidth=2, label= "Fitted_line")

plt.xlabel("Tv_Spent ($thousand)")
plt.ylabel("Sales")
plt.title("Simpel linear regression : Tv Spent vs Sales")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/03_simple_regression_line.png", dpi=120)
plt.close

# Multi linear Regression 

print("\n" + "=" *49)
print("multi linear regression  model all modes of ads to Sales")
print("-" *50)

model_multi = LinearRegression()
model_multi.fit( X_train_m,  y_train )

print("learned coeeficent")

for feature, coef in zip(x_multi.columns, model_multi.coef_):
    print(f"{feature:18s}: {coef:.4f}")
print(f"{'Intercept':18s}: {model_multi.intercept_:.4f}")

pred_multi = model_multi.predict (X_test_m)
mae_m = mean_absolute_error(y_test, pred_multi)
rmse_m = np.sqrt(mean_squared_error(y_test, pred_multi))
r2_m = r2_score(y_test, pred_multi)

print(f"test preformance")
print(f"MAE {mae_m:.3f}")
print(f" RMSE {rmse_m:.3f}")
print(f"R2 {r2_m:.3f}({r2_m*100:.1f}% of Sales varation)")

# Compare two model
print("\n" + "-" *59)
print(" compare simple vs multi")
print("="*60)

comparison = pd.DataFrame({
    "Model" : ["Simple (Tv only)", "Multi (all channels)"],
    "MAE" : [mae_s, mae_m],
    "RMSE" : [rmse_s, rmse_m],
    "R2" : [r2_s, r2_m]

}).round(3)

print(comparison.to_string(index=False))

plt.figure(figsize=(6,6))
plt.scatter(y_test, pred_multi, alpha=0.7, color="#534AB7")
lims = [min(y_test.min(), pred_multi.min()), max(y_test.max(), pred_multi.max())]
plt.plot(lims, lims, color="#888780", linestyle="--", label="prefect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("actual vs predicted (multi-model)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_actual_vs_predicted.png", dpi=120)
plt.close()

residuals = y_test - pred_multi
plt.figure(figsize=(7, 5))
plt.scatter(pred_multi, residuals, alpha=0.7, color="#D85A30")
plt.axhline(0, color="#444441", linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot - should look like random scatter around 0")
plt.tight_layout()
plt.savefig(f"{OUT}/05_residual_plot.png", dpi=120)
plt.close()

joblib.dump(model_multi, f"{OUT}/sales_regression_model.pkl")
print(f"\nModel saved to {OUT}/sales_regression_model.pkl")
print("\nAll plots saved to:", OUT)
print("\nDone.")
