 #neccessary libraries
import pandas as pd
# deployment
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data=pd.read_csv("spam.csv")
print(data.shape)
#cleaned data
data.drop_duplicates(inplace=True)#when performing operations directly on dataset
data['Category']=data['Category'].replace(['ham','spam'],['Not Spam','Spam'])
print(data.shape)
#checking missing vales
print(data.isnull().sum())


mess=data['Message']
cat=data['Category']
#splitting data to train and test
(mess_train,mess_test,cat_train,cat_test)=train_test_split(mess,cat,test_size=0.2)

#using CountVectorizer to convert categorical to numeric
cv=CountVectorizer(stop_words='english') 
features=cv.fit_transform(mess_train)

#creating model
model=MultinomialNB()
model.fit(features,cat_train)

## Test our model
features_test=cv.transform(mess_test)
print(model.score(features_test,cat_test))
def predict(message):
#predict data
       input_message=cv.transform([message]).toarray()
       result=model.predict(input_message)
       return result
st.header('Spam Detector')



input_mess=st.text_input("Enter Message Here")

if st.button('Validate'):
       output=predict(input_mess)
       st.markdown(output)