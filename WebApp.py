# Description: web app to predict stroke probability based on some factors as sex, age, etc.

# import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import streamlit as st
from imblearn.over_sampling import SMOTE

# Create a title and a sub-title
st.write("""
# Prognoza riscului aparitiei AVC-ului
Identificarea probabilitatii aparitiei AVC-ului utilizand învățarea autromată și python
""")

# Set a sub-header
st.subheader('Date generale')

# Open and display an image
image = Image.open('/Users/etutunaru/PycharmProjects/avc_app/AVC.png')
st.image(image)

st.write('Accidentul vascular cerebral este un tip de eșec brusc al fluxului normal de sânge în creier, rezultat din '
         'blocarea sau ruperea vaselor de sânge și se caracterizează printr-o apariție bruscă (în câteva minute, '
         'mai rar ore) a simptomelor neurologice focale (tulburări motorii, ale vorbirii, senzoriale, de coordonare, '
         'vizuale și de altă natură) și/sau tulburări cerebrale generale (deprimare a conștiinței, cefalee, '
         'vărsături etc.) care persistă mai mult de 24 de ore sau duc la moartea pacientului într-o perioadă scurtă '
         'de timp ca urmare a unei cauze de origine cerebrovasculară.')

st.write('Regresia logistică este un algoritm de clasificare în învățarea automată care este utilizat pentru a '
         'prezice probabilitatea unei variabile dependente categorice. În regresia logistică, variabila dependentă '
         'este o variabilă binară care conține date codificate ca 1 (da, succes etc.) sau 0 (nu, eșec etc.). Cu alte '
         'cuvinte, modelul de regresie logistică prezice P (Y = 1) în funcție de X.')

# Get the data
data = pd.read_csv('/Users/etutunaru/PycharmProjects/avc_app/cleaned_stroke_data.csv')

# Split the data into independent 'X' and dependent 'Y' variables
X = data.drop(['Id', 'AVC'], axis=1)
Y = data['AVC']

# Split the data set into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

# Data augmentation
sampler = SMOTE(random_state=42)
X_train_over_sm, y_train_over_sm = sampler.fit_resample(X_train, y_train.values.ravel())
y = pd.DataFrame({'AVC': y_train_over_sm})

st.sidebar.header('Formular')


# Get the feature input from the user
def get_user_input():
    sex_label = st.sidebar.radio('Sex:', ('Masculin', 'Feminin'))
    if sex_label == 'Masculin':
        sex = 0
    else:
        sex = 1

    age = st.sidebar.slider('Vârsta:', 18, 120, 40)

    hypertension_label = st.sidebar.slider('Presiunea sistolica:', 60, 280, 120)
    if hypertension_label > 139:
        hypertension = 1
    else:
        hypertension = 0

    heart_diseases_label = st.sidebar.radio('Boli de inimă:', ('Da', 'Nu'))
    if heart_diseases_label == 'Da':
        heart_diseases = 1
    else:
        heart_diseases = 0

    married_label = st.sidebar.radio('Vreodata căsătorit:', ('Da', 'Nu'))
    if married_label == 'Da':
        married = 1
    else:
        married = 0

    work_type_label = st.sidebar.radio('Tip de lucru:', ('Privat', 'Liber profesionist', 'Angajat la stat', 'Șomer'))
    if work_type_label == 'Angajat la stat':
        work_type = 0
    elif work_type_label == 'Somer':
        work_type = 1
    elif work_type_label == 'Privat':
        work_type = 2
    else:
        work_type = 3

    location_label = st.sidebar.radio('Reședință:', ('Rural', 'Urban'))
    if location_label == 'Urban':
        location = 1
    else:
        location = 0

    glucose = st.sidebar.slider('Glucoza:', 50, 300, 108)
    height = st.sidebar.slider('Înălțimea:', 120, 220, 171)
    weight = st.sidebar.slider('Greutatea:', 40, 300, 70)
    bmi = weight / ((height / 100) ** 2)
    smoke_label = st.sidebar.radio('Fumat:', ('Fost fumător', 'Nefumător', 'Fumător'))
    if smoke_label == 'Fost fumător':
        smoke = 1
    elif smoke_label == 'Nefumător':
        smoke = 2
    else:
        smoke = 3

    # Stroke a dictionary into a variable
    user_data = {
        'sex': sex,
        'varsta': age,
        'hipertensiune': hypertension,
        'boli de inima': heart_diseases,
        'vreodata casatorit': married,
        'tip de lucur': work_type,
        'resedinta': location,
        'glucoza': glucose,
        'IMC': bmi,
        'fumat': smoke
    }

    nominal_data = {
        'Sex': sex_label,
        'Vârsta': age,
        'Hipertensiune': hypertension_label,
        'Boli de inimă': heart_diseases_label,
        'Vreodata căsatorit': married_label,
        'Tip de lucur': work_type_label,
        'Reședința': location_label,
        'Glucoza': glucose,
        'IMC': bmi,
        'Fumat': smoke_label
    }

    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    features_nominal = pd.DataFrame(nominal_data, index=[0])
    return features, features_nominal


# Store the user input into a variable
user_input, user_nominal_input = get_user_input()

# Set a sub-header and display the users input
st.subheader('Datele utilizatorului:')
st.write(user_nominal_input)

# Create and train the model
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=400)
logisticRegr.fit(X_train_over_sm, y_train_over_sm)
prediction = logisticRegr.predict(X_test)

# Store the model prediction in a variable
user_prediction = logisticRegr.predict_proba(user_input)

# Display the result
st.subheader('Probabilitatea de a suferi AVC: ')
result = round(user_prediction[0][1] * 100, 2)
st.write(str(result) + '%')
