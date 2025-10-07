import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score


# 1. Load Data
df = pd.read_csv("../../data/taxifare.csv")

# 2. Inspect 

print(df.head())
print(df.info())

# 3. Clean data

df = df.dropna()
df = df[(df['TRIP_MILES'] > 0) & (df['FARE'] > 0)]
df = df[df['TRIP_MILES'] < 50]


#For simplicity, we'll use only 'trip_distance' as the feature and 'fare_amount' as the target
X= df[['TRIP_MILES']]
y= df['FARE']

# 4. Split into train/test

X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size=0.2 ,random_state=42)


# 5. Train Model

model = LinearRegression()
model.fit(X_train , y_train)


#6. Predict

y_pred = model.predict(X_test)

# 7. Evaluate

mse = mean_squared_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

m = model.coef_[0]
b = model.intercept_
print(f"Equation: fare = {m:.2f} * miles + {b:.2f}")

miles = 8.4
predicted_fare = model.predict([[miles]])[0]
print(f"For {miles} miles, predicted fare = {predicted_fare:.2f}")

# for size in [0.1, 0.2, 0.3, 0.5]:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Test size: {size}, MSE: {mse:.2f}")


# 8 . Visualize

plt.scatter(X_test , y_test , color='blue' , label = 'Actual')
plt.scatter(X_test , y_pred , color='red', alpha=0.5 , label = 'Predicted')
plt.xlabel('Trip Distance')
plt.ylabel('Fare Amount')
plt.legend()
plt.show()