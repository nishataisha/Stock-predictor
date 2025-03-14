from sklearn.preprocessing import MinMaxScaler, LabelEncoder #Importing Preprocessing Tools
from sklearn.model_selection import train_test_split #Tool to Split the Data into Training and Testing 
from sklearn.neighbors import KNeighborsClassifier #Importing our Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Used to see data results

df = pd.read_csv('Workshop 2/supervised_learning/data.csv') #df stands for DataFrame

print("Rows and Columns:", df.shape) #Get total number of rows and columns
print("-" * 100)


print("Columns of Our Data:", df.columns)

print("-" * 100)

#Removing Unnecessary Data
columns_removed = ['customerID', 'Gender', 'SeniorCitizen', 
                   'Partner', 'Dependents', 'InternetService', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod'] #These are the columns that we need to remove

df.drop(columns_removed, axis= 1, inplace= True)

print("New Rows and Columns:", df.shape) 
print("Columns of Our Data after Dropping Columns:", df.columns)
print("-" * 100)

categorical_features = [] 

for column in df:
    if df[column].dtype == object:
        categorical_features.append(column)

print("Categorical Columns:", categorical_features)

label_encoder = LabelEncoder()

for column in categorical_features:
    df[column] = label_encoder.fit_transform(df[column])

print("Columns after Enconding:", df[categorical_features])
print("-" * 100)

minmax_scaler = MinMaxScaler()

scaled_data = minmax_scaler.fit_transform(df) #Calculate and Scale the new set of data with a common range (from to 0 to 1)

scaled_df = pd.DataFrame(scaled_data, columns= df.columns) #Create a new DataFrame with the scaled data

print("New Scaled DataFrame after Scaling the data", scaled_df)
print("-" * 100)

x = scaled_df.drop(columns= ['Churn']) #Features
y = scaled_df['Churn'] #Target Column


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 42)

#In this case, we are using 5 neighbors to our model
knn = KNeighborsClassifier(n_neighbors= 5, weights= 'distance', metric= 'euclidean')

knn.fit(x_train, y_train) #Train the Data

y_predict = knn.predict(x_test) #Predict the Testing data

#Geting Accuracy of our Model
points = 0
size = len(y_test)
for index in range(size):
    if y_test.iloc[index] == y_predict[index]: 
        points += 1

accuracy = points / size 

print(f"Accuracy of our KNN model: {accuracy * 100:.2f}%")
print("-" * 100)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_predict)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap= "Purples")
plt.title("Confusion Matrix of KNN Model")
plt.show()