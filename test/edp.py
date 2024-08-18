import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data=pd.read_csv('C:/Users/ps713/OneDrive/Desktop/test/heart.csv')
print(data)

print(data.info())

data['Sex'] = data['Sex'].replace({'M': 1, 'F': 0})
print(data)

print(data.shape)

print(data['RestingECG'].unique())

data['RestingECG'] = data['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2})
print(data)

data['ChestPainType'].unique()

data['ChestPainType']= data['ChestPainType'].replace({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
print(data)

data['ExerciseAngina'].unique()

data['ExerciseAngina'] = data['ExerciseAngina'].replace({'N': 0, 'Y': 1})
print(data)

data['ST_Slope']=data['ST_Slope'].replace({'Up':0,'Flat':1,'Down':2})
print(data)

print(data.info())

from sklearn.model_selection import train_test_split

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier

model_Classifier = RandomForestClassifier()

model_Classifier.fit(X_train,y_train)

df = model_Classifier.predict(X_test)

def categorical(df):
  return ['Heart Disease' if x == 1 else 'No Heart Disease' for x in df]

result = categorical(df)
print(result)

model_Classifier.score(X_test,y_test)
# Streamlit app
def main():
    st.title("Heart Disease Prediction")

    st.write("This is a simple web app to predict heart disease using a machine learning model.")

    # Load and display data
    data = pd.read_csv('C:/Users/ps713/OneDrive/Desktop/test/heart.csv')
    if st.checkbox("Show Data"):
        st.write(data.head())

    # Train the model
    # model, accuracy = build_model()

    # st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    st.write("### Enter the patient's data to predict heart disease:")

    # User inputs
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.selectbox("ChestPainType", options=[0, 1, 2, 3])
    trestbps = st.number_input("RestingBP", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("FastingBS > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("RestingECG", options=[0, 1, 2])
    thalach = st.number_input("MaxHR", min_value=70, max_value=210, value=150)
    exang = st.selectbox("ExerciseAngina", options=[0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox("ST_Slope", options=[0, 1, 2])
    

    # Make prediction
    user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]]
    prediction = model_Classifier.predict(user_data)

    if st.button("Predict"):
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        st.write(f"Prediction: **{result}**")

if __name__ == "__main__":
    main()
