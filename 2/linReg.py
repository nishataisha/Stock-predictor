import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#Using TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

#Load Data
data = pd.read_csv('data (1).csv')
df = pd.DataFrame(data)

print("Original Shape:", df.shape)


#Handle price outliers
Q1_price = df['price'].quantile(0.10)
Q3_price = df['price'].quantile(0.90)
IQR_price = Q3 - Q1

outlier = []

for index,price in df['price'].items():
    if price>=(Q3_price+1.5*IQR_price) or price <=(Q1_price - 1.5*IQR_price):
        outlier.append(index)

#Handle area outliers
Q1_area = df['area'].quantile(0.25)
Q3_area = df['area'].quantile(0.75)
IQR_area = Q3_area - Q1_area

for index, area in df['area'].items():
    if area >= (Q3_area + 1.5 * IQR_area) or area <= (Q1_price - 1.5 * IQR_price):
        outlier.append(index)

#Remove outliers
df.drop(outlier, inplace=False)
print("Shape after Removing the Outliers:", df.shape)

if 'furshingstatus' in df.columns:
    df.drop(columns= ['furnishingstatus'], axis= 1)
print("Shape after removing unnecessary Column(s):", df.shape)

#Encoding categorical variables
categorical_columns = []
for column in df:
    if df[column].dtype == object:
        categorical_columns.append(column)

le = LabelEncoder()
for categorical in categorical_columns:
    df[categorical] = le.fit_transform(df[categorical])

df['bed_bath']=df['bedrooms']*df['bathrooms']

#Splitting data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=100)

#Feature Scaling
scaler = StandardScaler()
scale_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']#Scale columns to put in same range
df_train[scale_columns] = scaler.fit_transform(df_train[scale_columns])
df_test[scale_columns] = scaler.transform(df_test[scale_columns])

#Model Training
model = LinearRegression()
x_train = df_train.drop(columns=['price'])
y_train = df_train['price']
model.fit(x_train, y_train)

#Model Testing
x_test = df_test.drop(columns=['price'])
y_test = df_test['price']
y_prediction = model.predict(x_test)

#Evaluation
r2 = r2_score(y_test, y_prediction)
print("r^2 Score =", r2)

mse = mean_squared_error(y_test, y_prediction)
print("Mean Squared Error =", mse)

#Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_prediction, alpha=0.5, color='blue', label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Perfect Fit Line")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()