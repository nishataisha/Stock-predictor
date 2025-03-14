import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv('C:\\Users\\Fareesa Meraj\\AI\\reg.py\\data (1).csv')

df = pd.DataFrame(data)


print("Original Shape:", df.shape)

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

outlier = []

for index, price in df['price'].items():
    if price >= (Q3 + 1.5*IQR) or price <= (Q1 - 1.5*IQR):
        outlier.append(index)

Q1 = df['area'].quantile(0.25)
Q3 = df['area'].quantile(0.75)
IQR = Q3 - Q1

for index, area in df['area'].items():
    if area >= (Q3 + 1.5*IQR) or area <= (Q1 - 1.5*IQR):
        outlier.append(index)

df.drop(outlier, axis= 0, inplace= True)

print("Shape after Removing the Outliers:", df.shape)

#Removing Unnecessary Columns
df.drop(columns= ['furnishingstatus'], axis= 1, inplace= True)
print("Shape after removing unnecessary Column(s):", df.shape)


categorical_columns = []

for column in df:
    if df[column].dtype == object:
        categorical_columns.append(column)

le = LabelEncoder()

for categorical in categorical_columns:
    df[categorical] = le.fit_transform(df[categorical]) #This will turn the words into numbers

df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 100)


def init_params(n_x):
    hidden_units = 12  
    W1 = np.random.rand(hidden_units, n_x) - 0.5  
    b1 = np.random.rand(hidden_units, 1) - 0.5     
    W2 = np.random.rand(1, hidden_units) - 0.5     
    b2 = np.random.rand(1, 1) - 0.5                
    return W1, b1, W2, b2

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)  # Prevents overflow
    return 1 / (1 + np.exp(-Z))

def deriv_sigmoid(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def forward_prop(W1, b1, W2, b2, X):
    """
    Forward propagation:
    - X has shape (num_features, num_samples)
    - Z1 = W1.dot(X) + b1 will have shape (12, num_samples)
    """
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)  # Hidden layer
    Z2 = W2.dot(A1) + b2  # Output layer
    A2 = Z2  # Raw output
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]  # number of samples
    dZ2 = A2 - Y  # Derivative of the loss function
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_sigmoid(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return A2

def get_mse(predictions, Y):
    mse = mean_squared_error(Y.flatten(), predictions.flatten())
    print(f"Mean Squared Error: {mse:.2f}")
    return mse

def gradient_descent(X, Y, alpha, iterations):
    n_x = X.shape[0]  # Number of features
    W1, b1, W2, b2 = init_params(n_x)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration {i}:")
            predictions = get_predictions(A2)
            get_mse(predictions, Y)
    return W1, b1, W2, b2


x_train = df_train.drop(columns=['price'], axis=1)
y_train = df_train['price']

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)

y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

X = x_train.T  
Y = y_train.reshape(1, -1)



W1,b1,W2,b2 = gradient_descent(X, Y, alpha=0.8, iterations=8000)


x_test = df_test.drop(columns=['price'], axis=1)
y_test = df_test['price']


x_test = scaler_x.transform(x_test)  
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()  

# Transpose so that:
X_test = x_test.T  
Y_test = y_test.reshape(1, -1)  


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + b2
    Y_pred = Z2  # Linear activation for regression
    return Y_pred

Y_pred_test = forward_propagation(X_test, W1, b1, W2, b2)




mse_test = np.mean((Y_pred_test - Y_test) ** 2)
print(f"Test Mean Squared Error: {mse_test:.2f}")


y_test_original = scaler_y.inverse_transform(Y_test.T)  
y_pred_original = scaler_y.inverse_transform(Y_pred_test.T)  


r2 = r2_score(y_test_original, y_pred_original)
print(f"R^2 Score: {r2:.4f}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.5, color='blue', label="Predicted vs Actual")
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         color='red', linestyle='--', label="Perfect Fit Line")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()