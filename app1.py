



import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/tailo/OneDrive/Desktop/diabetes_prediction/diabetes.csv")

st.title('Diabetes Checkup')

st.subheader('Training Data')
st.write(df.describe())

st.subheader('Visualisation')
st.bar_chart(df)


x = df.drop(['Outcome'],axis = 1)
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =0)

def user_report():
  Pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  Glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  BloodPressure = st.sidebar.slider('BloodPressure', 0,122, 70 )
  SkinThickness = st.sidebar.slider('SkinThickness', 0,100, 20 )
  Insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  BMI = st.sidebar.slider('BMI', 0,67, 20 )
  DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.0,2.4, 0.47 )
  Age = st.sidebar.slider('Age', 21,88, 33 )
  
  user_report = {
      'Pregnancies':Pregnancies,
      'Glucose':Glucose,
      'BloodPressure':BloodPressure,
      'SkinThickness':SkinThickness,
      'Insulin':Insulin,
      'BMI':BMI,
      'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
      'Age':Age
  }
  report_data = pd.DataFrame(user_report, index=[0])
  return report_data

user_data = user_report()


rf  = RandomForestClassifier()
rf.fit(x_train, y_train)                                                                  

st.subheader('Train Data Accuracy: ')
st.write(str(accuracy_score(y_train, rf.predict(x_train-0.7))*100)+'%')

st.subheader('Test Data Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

user_result = rf.predict(user_data)

st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
  
st.write(output)




























