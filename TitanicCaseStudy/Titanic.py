import math 
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitianicCase():

    titanic_data = pd.read_csv('Titanic.csv')

    print("First five entries of the dataset")
    print(titanic_data.head())

    print("Number of the passengers:" + str(len(titanic_data)))

    print("Visualisation: Survived and Non-survived Passengers")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target).set_title("Survived and non-survived passenger")
    show()

    print("Visualisation: Survived and non-survived Gender")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target, hue="Sex").set_title("Survived and non-survived gender")
    show()

    print("Visualisation: Survived and non-survived Class")
    figure()
    target = "Pclass"

    countplot(data=titanic_data, x=target, hue="Pclass").set_title("Survived and non-survived Class")
    show()

    print("Visualisation: Survived and non-survived Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non-survived based on Age")
    show()

    print("Visualisation: Survived and non-survived Fares")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non-survived based on Fares")
    show()

    print("first 5 entries after removing zero column")
    print(titanic_data.head(5))

    print("Values of sex column")
    print(pd.get_dummies(titanic_data['Sex']))

    print("Value of sex column after removing zero field")
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    print(Sex.head(5))

    print("Value of Pclass column after removing zero field")
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    print(Pclass.head(5))

    print("Value of data after concatenating new columns")
    titanic_data = pd.concat([titanic_data, Sex, Pclass], axis=1)
    print(titanic_data.head(5))

    print("Values of data after removing the irrelevant contents")
    titanic_data.drop(["sibsp", "Parch", "Embarked", "Sex"], axis=1, inplace=True)
    print(titanic_data.head(5))

    # Ensure all column names are strings
    titanic_data.columns = titanic_data.columns.astype(str)

    x = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

    lgModel = LogisticRegression(max_iter=1000)

    lgModel.fit(xtrain, ytrain)

    prediction = lgModel.predict(xtest)

    print("Classification report of Logistic Regression is: ")
    print(classification_report(ytest, prediction))

    print("Confusion matrix of Logistic Regression is: ")
    print(confusion_matrix(ytest, prediction))

    print("Accuracy of Logistic Regression is: ")
    print(accuracy_score(ytest, prediction))

def main():
    print("---------Jay Infosytems-------")

    print("_______SUPERVISED MACHINE LEARNING________")

    print("________LOGISTIC REGRESSION OF TITANIC DATASET________")

    TitianicCase()

if __name__ == "__main__":
    main()
