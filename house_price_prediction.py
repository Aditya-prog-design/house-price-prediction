import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Dataset
data = {
    'Square_Feet': [1500, 2000, 1200, 1800, 2500, 2200, 1600, 3000],
    'Bedrooms': [3, 4, 2, 3, 4, 4, 3, 5],
    'Bathrooms': [2, 3, 1, 2, 3, 3, 2, 4],
    'Price': [250000, 350000, 180000, 300000, 450000, 400000, 270000, 550000]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Square_Feet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Test Prediction
sample_house = pd.DataFrame({
    'Square_Feet': [2200],
    'Bedrooms': [4],
    'Bathrooms': [3]
})

prediction = model.predict(sample_house)

print("Predicted House Price:", round(prediction[0], 2))

# Model Evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:", round(r2, 4))
print("Mean Squared Error:", round(mse, 2))

# Graph
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# User Input Prediction
print("\nEnter House Details")

sqft = float(input("Enter Square Feet: "))
bed = int(input("Enter Bedrooms: "))
bath = int(input("Enter Bathrooms: "))

new_house = pd.DataFrame({
    'Square_Feet': [sqft],
    'Bedrooms': [bed],
    'Bathrooms': [bath]
})

prediction = model.predict(new_house)

print("\nPredicted House Price = ₹{:,.2f}".format(prediction[0]))