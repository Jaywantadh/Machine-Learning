from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def get_data():
    data = datasets.load_wine()

    print(data.feature_names)

    print(data.target_names)

    print(data.data[0:5])

    print(data.target)

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def main():
    print("______JAY INFOSYSTEMS_______")
    print("Machine Learning Application")
    print("Wine predictor application using K-Nearest Knighbor algorithm")

    get_data()

if __name__ == "__main__":
    main()
    


