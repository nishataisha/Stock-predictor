from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import plot_tree 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_score 


df = pd.read_csv('C:\Users\Fareesa Meraj\AI\knn.py\knn\data.csv') 

print("Rows and Columns:", df.shape) 
print("-" * 100)


print("Columns of Our Data:", df.columns)

print("-" * 100)

columns_removed = ['customerID', 'Gender', 'SeniorCitizen', 
                   'Partner', 'Dependents', 'InternetService', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod'] 

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


x = df.drop(columns= ['Churn']) 
y = df['Churn'] 


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 42)



clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 4, min_samples_split= 2,min_samples_leaf= 2,
                            max_features= df.shape[1]) 

clf.fit(x_train, y_train) 

y_predicted = clf.predict(x_test) 


accuracy = precision_score(y_test, y_predicted) 

print(f"Accuracy of our Decision Tree model: {accuracy * 100:.2f}%")
print("-" * 100)



plt.figure(figsize=(25,15))
plot_tree(clf, filled= True, feature_names= df.columns, class_names= ['No Churn', 'Churn'], 
          rounded=True, proportion=False)

plt.title("Decision Tree Classifier Visualization")
plt.show()') #df stands for DataFrame

print("Rows and Columns:", df.shape) 
print("-" * 100)


print("Columns of Our Data:", df.columns)

print("-" * 100)

columns_removed = ['customerID', 'Gender', 'SeniorCitizen', 
                   'Partner', 'Dependents', 'InternetService', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod'] 

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


x = df.drop(columns= ['Churn']) #Features
y = df['Churn'] #Target Column


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 42)


clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 4, min_samples_split= 2,min_samples_leaf= 2,
                            max_features= df.shape[1]) 
clf.fit(x_train, y_train) #Train

y_predicted = clf.predict(x_test) #Predict Testing Data


accuracy = precision_score(y_test, y_predicted) #Compare predictions with the actual data

print(f"Accuracy of our Decision Tree model: {accuracy * 100:.2f}%")
print("-" * 100)

plt.figure(figsize=(25,15))
plot_tree(clf, filled= True, feature_names= df.columns, class_names= ['No Churn', 'Churn'], 
          rounded=True, proportion=False)

plt.title("Decision Tree Classifier Visualization")
plt.show()