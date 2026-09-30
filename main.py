import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("titanic_train.csv")
df["Age1"] = df["Age"]
df.loc[(df["Age1"].isnull()) & (df["Pclass"] == 1) & (df["Sex"] == "male"), "Age1"] = 41.28138613861386
df.loc[(df["Age1"].isnull()) & (df["Pclass"] == 2) & (df["Sex"] == "male"), "Age1"] = 30.74070707070707
df.loc[(df["Age1"].isnull()) & (df["Pclass"] == 3) & (df["Sex"] == "male"), "Age1"] = 26.507588932806325
df.loc[(df["Age1"].isnull()) & (df["Pclass"] == 1) & (df["Sex"] == "female"), "Age1"] = 41.28138613861386
df.loc[(df["Age1"].isnull()) & (df["Pclass"] == 2) & (df["Sex"] == "female"), "Age1"] = 30.74070707070707
df.loc[(df["Age1"].isnull()) & (df["Pclass"] == 3) & (df["Sex"] == "female"), "Age1"] = 26.507588932806325
df["Age"] = df["Age1"]
df = df.drop('Age1', axis=1)
df = df.drop('Cabin', axis=1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df.loc[df["Age"] <= 22, "Age"] = 0
df.loc[(df["Age"] > 22) & (df["Age"] <= 27), "Age"] = 1
df.loc[(df["Age"] > 27) & (df["Age"] <= 37), "Age"] = 2
df.loc[(df["Age"] > 37) & (df["Age"] <= 80), "Age"] = 3
df["Age"] = df["Age"].apply(lambda x: int(x))
df1 = df.drop(["Name", "SibSp", "Parch", "Ticket" ,"Fare", "Embarked", "PassengerId"], axis=1)
X = df1.drop("Survived", axis=1)
Y = df1["Survived"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2, random_state=101)
lr = LogisticRegression()
lr.fit(Xtrain, Ytrain)
Ypred = lr.predict(Xtest)



st.text("hello")

Age = st.slider("How old are you?", 0, 130, 25)
st.write("I'm ", Age, "years old")


Sex = ["Male", "Female"]
Sex_selection = st.segmented_control(
    "Directions", Sex, selection_mode="single"
)

Class = ["1st class", "2nd class", "3rd class"]
Class_selection = st.segmented_control(
    "Directions", Class, selection_mode="single"
)
def predict():   
    class_map = {"1st class": 1, "2nd class": 2, "3rd class": 3}
    pclass = class_map.get(Class_selection, 3)

    sex_map = {"Male": 0, "Female": 1}
    sex = sex_map.get(Sex_selection, 0)

    if Age <= 22:
        age_bin = 0
    elif Age <= 27:
        age_bin = 1
    elif Age <= 37:
        age_bin = 2
    else:
        age_bin = 3

    features = [[pclass, sex, age_bin]]
    
    return lr.predict(features)[0]


if st.button("Done", type="primary"):
    if int(predict()) == 1:
        st.text("you survived!")
    else:
        st.text("You didn't survive.")