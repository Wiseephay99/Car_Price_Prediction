import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
df = pd.read_csv('car_cleaned_data.csv')
df.head()
df.columns
X = df[['Present_Price', 'Kms_Driven', 'Car_Age', 'Fuel_Type_CNG', 'Fuel_Type_Diesel',
        'Fuel_Type_Petrol', 'Transmission_Automatic', 'Transmission_Manual']]

y = df['Selling_Price']

# Fittinng Simple Linear Regression Model

model = LinearRegression()
model.fit(X, y)

with open('lr_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as lr_model.pkl")
