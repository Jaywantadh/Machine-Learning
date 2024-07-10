import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib as plot

def AdvertismentPredictor(data_path):
    data = pd.read_csv(data_path, index_col=0)

    print("Size of features", len(data))
    feature_names = ['TV', 'radio', 'newspaper']

    print("Names of feature", feature_names)

    X = data[feature_names]
    Y = data.sales

    X_train, X_test, Y_test, Y_train = train_test_split(X,Y, test_size=1/2)

    print("Size of Training dataset", len(X_train))
    print("Size of Testing Dataset", len(X_test))

    linreg = LinearRegression()

    linreg.fit(X_train, Y_train)

    y_pred = linreg.predict(X_test)

    print("Testing set")
    print(X_test)

    print("Results of testing: ")
    print(y_pred)

    print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

def main():
    AdvertismentPredictor("Advertising .csv")

if __name__ == "__main__":
    main()
