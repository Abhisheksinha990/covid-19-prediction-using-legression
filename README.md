import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and clean data
data = pd.read_csv(r"C:\Users\Abhishek sinha\Downloads\country_wise_latest.csv")
data = data.dropna()

# Pairplot for Numeric Features
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.pairplot(data.select_dtypes(include='number'), diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = data.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

# Regression: Confirmed vs Deaths
x1 = data['Confirmed']
y1 = data['Deaths']
slope1, intercept1 = np.polyfit(x1, y1, 1)
r_squared1 = np.corrcoef(x1, y1)[0, 1] ** 2
plt.figure(figsize=(8, 5))
sns.regplot(x=x1, y=y1, line_kws={"color": "red"})
plt.title(f'Regression: Confirmed vs Deaths\n$R^2$ = {r_squared1:.2f}')
plt.xlabel('Confirmed Cases')
plt.ylabel('Deaths')
plt.grid(True)
plt.tight_layout()
plt.show()

# Regression: Confirmed vs Recovered
x2 = data['Confirmed']
y2 = data['Recovered']
slope2, intercept2 = np.polyfit(x2, y2, 1)
r_squared2 = np.corrcoef(x2, y2)[0, 1] ** 2
plt.figure(figsize=(8, 5))
sns.regplot(x=x2, y=y2, line_kws={"color": "blue"})
plt.title(f'Regression: Confirmed vs Recovered\n$R^2$ = {r_squared2:.2f}')
plt.xlabel('Confirmed Cases')
plt.ylabel('Recovered Cases')
plt.grid(True)
plt.tight_layout()
plt.show()

# Regression: Deaths vs Deaths per 100 Cases
x3 = data['Deaths']
y3 = data['Deaths / 100 Cases']
slope3, intercept3 = np.polyfit(x3, y3, 1)
r_squared3 = np.corrcoef(x3, y3)[0, 1] ** 2
plt.figure(figsize=(8, 5))
sns.regplot(x=x3, y=y3, line_kws={"color": "green"})
plt.title(f'Regression: Deaths vs Deaths per 100 Cases\n$R^2$ = {r_squared3:.2f}')
plt.xlabel('Deaths')
plt.ylabel('Deaths per 100 Cases')
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Regression: Active vs Deaths
x4 = data['Active']
y4 = data['Deaths']
slope4, intercept4 = np.polyfit(x4, y4, 1)
r_squared4 = np.corrcoef(x4, y4)[0, 1] ** 2
plt.figure(figsize=(8, 5))
sns.regplot(x=x4, y=y4, line_kws={"color": "purple"})
plt.title(f'Regression: Active vs Deaths\n$R^2$ = {r_squared4:.2f}')
plt.xlabel('Active Cases')
plt.ylabel('Deaths')
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Regression: Recovered vs Active
x5 = data['Recovered']
y5 = data['Active']
slope5, intercept5 = np.polyfit(x5, y5, 1)
r_squared5 = np.corrcoef(x5, y5)[0, 1] ** 2
plt.figure(figsize=(8, 5))
sns.regplot(x=x5, y=y5, line_kws={"color": "brown"})
plt.title(f'Regression: Recovered vs Active\n$R^2$ = {r_squared5:.2f}')
plt.xlabel('Recovered Cases')
plt.ylabel('Active Cases')
plt.grid(True)
plt.tight_layout()
plt.show()

# Distribution of Confirmed Cases
plt.figure(figsize=(8, 5))
sns.histplot(data['Confirmed'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Confirmed Cases')
plt.xlabel('Confirmed Cases')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Top 10 Countries by Confirmed Cases
top_confirmed = data.sort_values('Confirmed', ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(data=top_confirmed, x='Country/Region', y='Confirmed', palette='viridis')
plt.xticks(rotation=45)
plt.title('Top 10 Countries by Confirmed Cases')
plt.tight_layout()
plt.show()

# Scatter: Confirmed vs Active Cases
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Confirmed', y='Active', color='orange')
plt.title('Scatter: Confirmed vs Active Cases')
plt.xlabel('Confirmed')
plt.ylabel('Active')
plt.grid(True)
plt.tight_layout()
plt.show()

# Improved: Top 10 Countries by Deaths per 100 Cases
top10_deaths_per_100 = data.sort_values('Deaths / 100 Cases', ascending=False).head(10)
plt.figure(figsize=(12, 6))
bar = sns.barplot(
    data=top10_deaths_per_100,
    x='Deaths / 100 Cases',
    y='Country/Region',
    palette='Reds_r'
)
# Annotate each bar with value
for index, value in enumerate(top10_deaths_per_100['Deaths / 100 Cases']):
    bar.text(value + 0.2, index, f'{value:.2f}', color='black', va='center')

plt.title('Top 10 Countries by Deaths per 100 Cases', fontsize=14, weight='bold')
plt.xlabel('Deaths per 100 Confirmed Cases')
plt.ylabel('Country')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Linear Regression Model
X = data[['Confirmed']]
y = data['Deaths']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nRegression Model Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
