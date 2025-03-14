import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def randomForest_train_predict(X_train, Y_train, X_test):
    #Create Model
    model = RandomForestRegressor(random_state=42)

    #Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [10, 50, 100, 150],
        'max_depth': [None,10, 20, 30],
        'max_features':['log2', 'sqrt'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error']
    }
    #Search to get the best model
    gridSearch = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

    #Fit the model
    gridSearch.fit(X_train, Y_train)

    model = gridSearch.best_estimator_

    #Predict the Test Data
    predictions = model.predict(X_test)

    return predictions