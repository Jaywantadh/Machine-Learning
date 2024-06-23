import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictor(data_path):

    data = pd.read_csv(data_path)

    print("Size of actual dataset", len(data))

    features_names = ['Whether', 'Temperature']

    print("Names of Features", features_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    le = preprocessing.LabelEncoder()

    weather_encoded = le.fit_transform(whether)
    print(weather_encoded)

    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)

    print(temp_encoded)

    features = list(zip(weather_encoded, temp_encoded))

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(features, label)

    predicted = model.predict([[0,2]])
    print(predicted)

def main():
    print("-------Jay Infosystems-----")
    print("Machine Learning Application")
    print("Play Predictor application using K Nearest Neighbour Algorithm")

    PlayPredictor('PlayPredictor.csv')

if __name__ == "__main__":
    main()
