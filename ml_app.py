import streamlit as st
import pandas as pd
import numpy as np


st.title(" An app that predicts penguin species")

df=pd.read_csv("data/penguins.csv")
st.write(df.tail())
st.write(df['species'].unique())
st.write(df['island'].unique())
st.write(df['sex'].unique())

# remove the nulls
df.dropna(inplace=True)


# sepearate features and label
X=df.drop(['species', 'year'], axis=1)
y=df.species

# lets encopde the categorical variables
X=pd.get_dummies(X)
y, uniques=pd.factorize(y)

st.write(y)
st.write(uniques)

# make a train test split
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

# train a random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc=RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)
y_pred=rfc.predict(x_test)
score=accuracy_score(y_pred, y_test)
st.write(f"The accuracy score of the Random Forest Model is {score}")

# Serialize the model (pickle the model) and the mappings 

import pickle
model=open('rfc.pickle', 'wb')
pickle.dump(rfc, model)
model.close()

mapping=open('mapping.pickle', 'wb')
pickle.dump(uniques, mapping)
mapping.close()

#create a UI for user inputs 
island=st.selectbox('island', ['Biscoe', 'Dream', 'Torgersen'])
sex=st.selectbox('sex', ['female', 'male'])
bill_length=st.number_input('bill_length(mm)', min_value=0)
bill_depth=st.number_input('bill_depth(mm)', min_value=0)
flipper_length=st.number_input('flipper_length(mm)', min_value=0)
body_mass=st.number_input('body_mass(g)', min_value=0)

st.write("The user inputs are [island, sex, bill_length, bill_depth, flipper_length, flipper_depth, body_mass]")

# create a mapping for the categoricals
island_Biscoe, island_Dream, island_Torgerson=0,0,0
if island=='Biscoe':
    island_Biscoe=1
elif island=='Dream':
    island_Dream=1
elif island=='Torgerson':
    island_Torgerson=1
    
sex_female, sex_male=0,0
if sex=='female':
    sex_female=1
elif sex=='male':
    sex_male=1

# deploy the app
# open the model and mapping pickles
model=open('rfc.pickle', 'rb')
rfc=pickle.load(model)
model.close()

mapping=open('mapping.pickle', 'rb')
map=pickle.load(mapping)
mapping.close()
#st.write(X)
if st.button('Predict'):
    prediction=rfc.predict([[  bill_length, bill_depth, flipper_length, body_mass,island_Biscoe, island_Dream, island_Torgerson,
                           sex_female, sex_male]])
    species_predicted=map[prediction][0]
    st.success(f"The predicted species is {species_predicted}")
    
