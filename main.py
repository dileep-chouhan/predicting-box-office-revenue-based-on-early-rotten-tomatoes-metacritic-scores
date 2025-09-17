import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'RottenTomatoes': np.random.randint(30, 100, num_movies),
    'Metacritic': np.random.randint(30, 100, num_movies),
    'Budget': np.random.randint(1000000, 100000000, num_movies), #Budget in USD
    'BoxOfficeRevenue': np.random.randint(1000000, 1000000000, num_movies) #Box Office Revenue in USD
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for this synthetic data.  Real-world data would require more extensive cleaning.
# --- 3. Exploratory Data Analysis (EDA) ---
# Visualize relationships between variables
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.regplot(x='RottenTomatoes', y='BoxOfficeRevenue', data=df)
plt.title('Rotten Tomatoes vs. Box Office Revenue')
plt.subplot(1, 2, 2)
sns.regplot(x='Metacritic', y='BoxOfficeRevenue', data=df)
plt.title('Metacritic vs. Box Office Revenue')
plt.tight_layout()
plt.savefig('eda_plots.png')
print("Plot saved to eda_plots.png")
# --- 4. Model Building ---
# Simple linear regression model (can be expanded to more complex models)
X = df[['RottenTomatoes', 'Metacritic']]
y = df['BoxOfficeRevenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 6. Visualization of Model Performance ---
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Box Office Revenue")
plt.ylabel("Predicted Box Office Revenue")
plt.title("Actual vs Predicted Box Office Revenue")
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], color='red', linestyle='--') #Line of perfect prediction
plt.savefig('model_performance.png')
print("Plot saved to model_performance.png")
# --- 7.  Prediction for new movie (example)---
new_movie = pd.DataFrame({'RottenTomatoes': [85], 'Metacritic': [80]})
predicted_revenue = model.predict(new_movie)
print(f"Predicted Box Office Revenue for new movie: ${predicted_revenue[0]:,.2f}")