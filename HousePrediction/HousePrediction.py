import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Loading data by using pandas
data = pd.read_csv('Housing.csv')

# Getting the date vlaues in correct format
data['date'] = pd.to_datetime(data['date'])

# Getting year,month,day from the data
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# Drop the original date column
data = data.drop(['date'], axis=1)

x = data.drop('price', axis=1)
y = data['price']

# Identify categorical columns
categorical_features = x.select_dtypes(include=['object']).columns

# Create a preprocessor using ColumnTransformer and select the data which are not having categorial values and convert the categorial values into binary values using onehotencoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),x.select_dtypes(include=['number']).columns),
        ('cat',OneHotEncoder(),categorical_features)
    ],
    remainder = 'passthrough'
)

# Transform the data
x_scaled = preprocessor.fit_transform(x)

# Split the data to training and testing
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = Sequential()

# Add the input layer
model.add(Dense(units=64,activation='relu',input_dim=x_scaled.shape[1]))

# Adding the hidden layers
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=16,activation='relu'))

# Add output layer (1 unit for regression, no activation function)
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error')

# Training the model
history = model.fit(x_train,y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=0)

plt.figure(figsize=(5,5))
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
loss = model.evaluate(x_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

# Make predictions
predictions = model.predict(x_test)

# Plot actual vs predicted values with the best-fit regression line
plt.figure(figsize=(5,5))
plt.scatter(y_test, predictions, label='Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# Fit a linear regression line
regression_line = np.polyfit(y_test, predictions.flatten(), 1)
plt.plot(y_test, np.polyval(regression_line, y_test), color='red', label='Regression Line')
plt.title('Actual Prices vs Predicted Prices with Regression Line')
plt.legend()
plt.show()