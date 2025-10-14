import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # to convert text to numerical data
from sklearn.naive_bayes import MultinomialNB # Naive Bayes classifier
import streamlit as st

data = pd.read_csv('spam.csv') #reading the dataset

# print(data.head()) # Display the first few rows of the dataset
# print(data.shape) # Display the shape of the dataset (5572, 2)

data.drop_duplicates(inplace=True) # Remove duplicate entries
# print(data.shape) # Display the shape after removing duplicates
# print(data.isnull().sum()) # Check for missing values  (5157, 2)

data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam']) # Encode 'ham' as 0 and 'spam' as 1
# print(data.head())

mess = data['Message']
cat = data['Category']

(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2) # Split the dataset into training and testing sets

cv = CountVectorizer(stop_words='english') # Initialize CountVectorizer with English stop words
features = cv.fit_transform(mess_train) # Fit the CountVectorizer on the training messages

#creating the model

model = MultinomialNB() # Initialize the Multinomial Naive Bayes classifier
model.fit(features, cat_train) # Train the model

#testing the model

features_test = cv.transform(mess_test) # Transform the test messages using the fitted CountVectorizer
# print(model.score(features_test, cat_test)) # Evaluate the model on the test set and print the accuracy

#predicting the data
def predict(message):
  input_message = cv.transform([message]).toarray() # Transform a new message
  result = model.predict(input_message) # Predict the category of the new message
  return result
# print(result) # Print the prediction result
# print(predict("Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."))

st.header("Spam Detection Tester (By Sidman)") # Title of the web app
st.write("Type the email text body below to find out if it’s Spam or Not Spam!")
input_mess = st.text_area("Enter your message:", height=150)# Input box for user to enter a message

if st.button("Predict"): # Button to trigger prediction
  if input_mess.strip() == "":
        st.warning("Please enter a message before predicting.")
  else:
      input_features = cv.transform([input_mess])
      prediction = model.predict(input_features)[0]
      if prediction == "Spam":
          st.error("This message looks like SPAM! Be careful.")
      else:
          st.success("This message does not look like spam.")