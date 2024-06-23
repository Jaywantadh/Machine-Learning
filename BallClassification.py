from sklearn import tree

#This case study deals with features like surface of ball(Smooth, Rough) and from that surface it decides if the ball is Tennis ball or Cricket ball.
#Data is filled manually.

#BALL SURFACE
#rough 1
#smooth 0

#BALL TYPE
#Tennis 1
#Cricket 2



def MyClassifier(wheight, surface):
    # feature Encoding
    Features = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0], [92, 0], [35, 1], [35, 1], [35, 1]]

    # label Encoding
    Labels = [1, 1, 2, 1, 2, 2, 1, 1, 1]

    # Decide the Algorithm
    obj = tree.DecisionTreeClassifier()

    # train the mode
    obj = obj.fit(Features, Labels)

    # Test the model
    ret = print(obj.predict([[wheight, surface]]))
    if ret == 1:
        print("Your object looks like a Tennis ball")
    else:
        print("Your object looks like a Cricket ball")
def main():
    print("----------Ball Classifiication case study---------")

    print("Please enter the informartion about the objct that you want to test")

    print("Please enter wheight of your object in grams")
    no = int(input())

    data = input("Please enter the type of surface(smooth/rough): ")

    if data.lower() == "rough":
        data = 1
    elif data.lower() == "Smooth":
        data = 0
    else:
        print("Invalid input")
        exit()

    MyClassifier(no, data)

if __name__ == "__main__":
    main()
