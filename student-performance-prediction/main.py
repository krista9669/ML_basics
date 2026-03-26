# supervised machine learning -> regression
# input features and a known output, model learns relationship between them

# splitting -> scaling

import pandas as pd # data handling
import numpy as np  # numerical operations

from sklearn.model_selection import train_test_split    # Data splitting
from sklearn.preprocessing import StandardScaler    # Feature scaling
from sklearn.linear_model import LinearRegression   # Linear regression model
from sklearn.metrics import mean_absolute_error, mean_squared_error # Error evaluation

data = pd.read_csv("student.csv", sep=";")
print(data.head())
# read csv files and print first 5 rows

features = ["studytime", "absences", "G1", "G2"] # defines the features that the model would use as input
target = "G3" # final prediction

X = data[features]  # inputs
y = data[target]    # outputs

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42    #fixing randomness gives consistency
)
# this is for splitting the data -> for more efficient test and train

# scaling is used so that there is no bias and all the scaling features are comparable and balanced or on a similar scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # fit learns from the data, the average and the range of scale and rescales
X_test = scaler.transform(X_test) # not allowed to learn from it hence only transform -> not fit here so no memory leakages

model = LinearRegression()  # I want to use a linear regression model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # model’s guessed answers

mae = mean_absolute_error(y_test, y_pred)   # how far off are are my predictions, difference btw actual and predicted later average of those differences
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # how bad are my worst mistakes, difference btw actual and predicted, squares the difference then average of that, square_root(average) 

print("\nResults:")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))