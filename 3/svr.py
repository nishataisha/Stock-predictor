import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def svr_train_predict(X_train, Y_train, X_test):
    #Scale the data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    Y_train_shaped = scaler_Y.fit_transform(Y_train.values.reshape(-1, 1))

    #Create SVR model
    model = SVR()

    #Defining hyperparameters to tune
    param_grid = {
        'kernel': ['linear','rbf','sigmoid'],
        'C':[1,10],
        'gamma':['scale']
    }

    gridSearch = GridSearchCV(estimator = model, param_grid = param_grid, cv =3, n_jobs = -1, verbose =2)

    #Fit the model
    gridSearch.fit(X_train_scaled, Y_train_shaped.ravel())

    model=gridSearch.best_estimator_ #return the BEST model

    #predictions
    predictions_scaled = model.predict(X_test_scaled)
    #Inversing the scaling to org. values
    predictions = scaler_Y.inverse_transform(predictions_scaled.reshape(-1,1))

    return predictions