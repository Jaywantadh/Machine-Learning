import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def get_data(data):
    data = pd.read_csv(data)

    print("Size of dataset:", data.shape)

    X = data[['TV','radio','newspaper']].values
    Y = data['sales'].values

    Y = Y.reshape((-1,1))

    n = len(X)

    reg = LinearRegression()

    reg = reg.fit(X, Y)

    y_pred = reg.predict(X)

    r2 = reg.score(X, Y)

    print(r2)

def main():
    print("_______Jay infosystems_______")
    print("_______SUPERVISED LEARNING_______")
    print("Linear regression model of Advertising agency test case.")

    get_data('Advertising .csv')

if __name__ == "__main__":
    main()
