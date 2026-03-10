import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Melbourne House Price AI",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Melbourne House Price Prediction AI")
st.write("Machine Learning Web Application using Streamlit")


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("MELBOURNE_HOUSE_PRICES_LESS.csv")
df = df.dropna(subset=['Price'])

cols = ['Rooms','Distance','Propertycount','Postcode','Type','Price']
df_filtered = df[cols].copy()

df_filtered = pd.get_dummies(df_filtered, columns=['Type'], drop_first=True)
df_filtered = df_filtered.fillna(df_filtered.median())


# ===============================
# SPLIT DATA
# ===============================
X = df_filtered.drop('Price',axis=1)
y = df_filtered['Price']

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size=0.2,random_state=42)


# ===============================
# TRAIN MODEL
# ===============================
model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)


# ===============================
# MODEL METRICS
# ===============================
mae = metrics.mean_absolute_error(y_test,predictions)
r2 = metrics.r2_score(y_test,predictions)


# ==============================
# TABS
# ==============================
tab1, tab2, tab3 = st.tabs([
"🏡 Prediction Dashboard",
"📊 Data Analysis",
"📈 Model Insights"
])


# ==============================
# TAB 1 : PREDICTION
# ==============================
with tab1:

    st.header("Predict House Price")

    st.sidebar.header("Enter House Features")

    rooms = st.sidebar.slider("Rooms",1,10,3)
    distance = st.sidebar.slider("Distance from CBD (km)",0.0,50.0,10.0)
    propertycount = st.sidebar.slider("Property Count",0,5000,1000)
    postcode = st.sidebar.number_input("Postcode",3000)

    property_type = st.sidebar.selectbox(
    "Property Type",
    ["House","Unit","Townhouse"]
    )

    type_u = 1 if property_type == "Unit" else 0
    type_t = 1 if property_type == "Townhouse" else 0

    input_data = pd.DataFrame({
      'Rooms':[rooms],
      'Distance':[distance],
      'Propertycount':[propertycount],
      'Postcode':[postcode],
      'Type_t':[type_t],
      'Type_u':[type_u]
})

    input_data = input_data[X.columns]
    prediction = model.predict(input_data)

    st.subheader("Predicted House Price")

    st.success(f"${prediction[0]:,.2f}")


# ===============================
# TAB 2 : DATA ANALYSIS
# ===============================
with tab2:

    st.header("Dataset Overview")

    st.write(df.head())

    st.subheader("Price Distribution")

    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered['Price'], kde=True, ax=ax1)
    ax1.set_title("Distribution of House Prices")
    st.pyplot(fig1)

    st.subheader("Price vs Distance")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        data=df_filtered,
        x='Distance',
        y='Price',
        ax=ax2
    )
    ax2.set_title("Price vs Distance from CBD")
    st.pyplot(fig2)


# ===============================
# TAB 3 : MODEL INSIGHTS
# ===============================
with tab3:

    st.header("Model Performance")

    st.metric("Mean Absolute Error", f"${mae:,.2f}")
    st.metric("R² Score", f"{r2:.3f}")

    st.subheader("Features Used in Model")

    st.write(X.columns.tolist())

    st.write("Dataset Shape:", df_filtered.shape)