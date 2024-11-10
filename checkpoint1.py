#I-Install the necessary packages
import pandas as pd
import streamlit as st 
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# give a title to our app 
st.title('Streamlit checkpoint 1') 

#II-Import you data and perform basic data exploration phase
df = pd.read_csv('Expresso_churn_dataset.csv')

#1-Display general information about the dataset
    #Display the first lines
st.subheader("Display the first lines")    
st.write(df.head())
    #Data information
st.subheader("Data information")

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text(s)
    #Statistical summary of the dataset
st.subheader("Statistical summary of the dataset")
st.write(df.describe())


#3-Missing and corrupted values
st.subheader("Missing and corrupted values")
missing_values = df.isnull().sum()
st.write(missing_values)

#Handle missing values
st.subheader("Handle missing values")
df.dropna(inplace=True)
df1=df
st.write(df1)

#Data information
st.subheader("Data information")
st.write(df1.info())

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

#4-Duplicates values
st.subheader("Duplicates values")
duplicates = df1.duplicated().sum()
st.write(f"Number of duplicate values ​​in dataset: {duplicates}")

#5-Handle outliers, if they exist

#Show numeric columns
st.subheader("Numeric columns")
numeric_columns = df1.select_dtypes(include=['number']).columns
st.write (numeric_columns)

#Display box plots for numeric column
st.subheader("box plots for numeric column")
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df1[column])
    plt.title(f'Box Plot of {column}')
    plt.show()

st.pyplot(plt)

#6-Encode categorical features

#Show categorical features
st.subheader("categorial_columns")
categorial_columns = df1.select_dtypes(include=['object']).columns
st.write (categorial_columns)

#Show categorical value counts
st.subheader("categorial value columns")

user_id_counts = df1['user_id'].value_counts()
st.write (user_id_counts)

region_counts = df1['REGION'].value_counts()
st.write (region_counts)

Tenure_counts = df1['TENURE'].value_counts()
st.write (Tenure_counts)

MRG_counts = df1['MRG'].value_counts()
st.write (MRG_counts)

pack_counts = df1['TOP_PACK'].value_counts()
st.write (pack_counts)

#transform Tenure column into a numerical feature
st.subheader("Transforming the 'TENURE' column")
# Define a mapping for 'TENURE' values to numerical values
tenure_mapping = {
    'K > 24 month': 25,
    'I 18-21 month': 19,
    'G 12-15 month': 13.5,
    'J 21-24 month': 22.5, 
    'F 9-12 month': 10.5, 
    'E 6-9 month': 7.5,   
    'D 3-6 month': 4.5}

# Apply the mapping to the 'TENURE' column
df1['TENURE'] = df1['TENURE'].map(tenure_mapping)

st.subheader("DataFrame after transformation")
st.dataframe(df1)

#transform TOP_PACK column into a numerical feature
st.subheader("Transforming the 'TOP_PACK' column")

# Grouping similar packs
# Function to simplify the values ​​of 'TOP_PACK'
def simplify_top_pack(top_pack):
    # Check if it is a data pack
    if 'Data' in top_pack:
        return 'Data Pack'
    # Check if it is an pack On-net
    elif 'On net' in top_pack or 'On-net' in top_pack:
        return 'On-net Pack'
    # Check if it is an pack All-net
    elif 'All-net' in top_pack:
        return 'All-net Pack'
    # Check if it is an pack Mixt
    elif 'MIXT' in top_pack:
        return 'Mixt Pack'
    # # Check if it is an IVR pack or something else
    elif 'IVR' in top_pack or 'VAS' in top_pack:
        return 'IVR/Service Pack'
    # If no category is found, 'Other Pack' can be returned
    else:
        return 'Other Pack'

# Apply the function to simplify the 'TOP_PACK' column
df1['simplify_top_pack'] = df1['TOP_PACK'].apply(simplify_top_pack)

# Show results
st.subheader("simplify the 'TOP_PACK' column")
st.write(df1[['TOP_PACK', 'simplify_top_pack']])

# Mapping 'Simplified_Top_Pack' values ​​to numeric values
st.subheader("Mapping 'Simplified_Top_Pack' values ​​to numeric values")

top_pack_mapping = {
    'Data Pack': 1,
    'On-net Pack': 2,
    'All-net Pack': 3,
    'Mixt Pack': 4,
    'IVR/Service Pack': 5,
    'Other Pack': 6
}

# Apply the mapping to the 'TENURE' column
df1['TOP_PACK_NUM'] = df1['simplify_top_pack'].map(top_pack_mapping)

# Show results
st.write(df1[['TOP_PACK', 'simplify_top_pack', 'TOP_PACK_NUM']])

#Encode region column
st.subheader("Encode region column")
label_encoder = LabelEncoder()
df1['REGION_encoded']= label_encoder.fit_transform(df1['REGION'])

st.write(df1[['REGION','REGION_encoded']])

#Remove columns
st.subheader("Remove columns")
df_encoded = df1.drop(['REGION','TOP_PACK', 'user_id','MRG' ], axis=1)
st.write(df_encoded)

#III-Based on the previous data exploration train and test a machine learning classifier
st.subheader('Prepare data for the modelling phase')

#1-Select your target variable and the features
st.subheader('Select your target and features')
target = 'CHURN'
features=['REGION_encoded', 'TENURE','ORANGE', 'TIGO', 'ZONE1', 'ZONE2','TOP_PACK_NUM', 'REGULARITY']

st.write("variable X")
X=df_encoded[features]
st.dataframe(X)

st.write("variable y")
y=df_encoded[target]
st.dataframe(y)

#splitting data with test size of 30%
from sklearn.model_selection import train_test_split
st.write("splitting data with test size of 30%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.write("X_train")
st.write("X_test")
st.write("y_train")
st.write("y_test")

#Apply a random forest
st.subheader('Apply a random forest model')

#Creating a random forest
randomforest = RandomForestClassifier(n_estimators=50, random_state=42)

#Training the model
randomforest.fit(X_train, y_train)

#testing the model
y_pred = randomforest.predict(X_test)

# Checking the accuracy
accuracy = randomforest.score(X_test, y_test)
st.write("Random Forest model accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

#IV-Add input fields for your features and a validation button at the end of the form
st.title('Churn predect')

# Add numeric input fields
region_input = st.number_input("Enter the Region code", min_value=0, max_value=13)
tenure_input = st.number_input("Enter the Tenure",min_value=4, max_value=25 )
orange_input = st.number_input("Enter the ORANGE value (0 or 1)")
tigo_input = st.number_input("Enter the TIGO value (0 or 1)", min_value=0, max_value=1)
zone1_input = st.number_input("Enter the ZONE1 value (0 or 1)", min_value=0, max_value=1)
zone2_input = st.number_input("Enter the ZONE2 value (0 or 1)", min_value=0, max_value=1)
pack_input = st.number_input("Enter the Pack number (1-6)", min_value=1, max_value=6)
regularity_input = st.number_input("Enter the Regularity", min_value=0.0)

# Splitting the data with a test size of 30%
st.write("Splitting the data into training and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import and train a RandomForestClassifier

st.subheader('Training the RandomForest Classifier')

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Add a validation button
if st.button('Validate'):
    input_data= pd.DataFrame([[region_input, tenure_input, orange_input, tigo_input, zone1_input, zone2_input, pack_input, regularity_input]], columns=features)
        # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.write("Prediction: The customer is likely to churn (Yes).")
    else:
        st.write("Prediction: The customer is unlikely to churn (No).")


