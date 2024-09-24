import streamlit as st
import pickle
import json
import numpy as np
import plotly.express as px
import pandas as pd

# Load the model and column information
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Function to predict price
def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0],2)

# Function to create dummy data for plotting linear regression
def create_plot_data(location,bath=2,bhk=2):
    # Create sample data for sqft and price predictions
    sqft_values = np.linspace(500, 4000, 100)
    price_predictions = [predict_price(location, sqft, bath, bhk) for sqft in sqft_values]
    
    df = pd.DataFrame({
        'Square Feet': sqft_values,
        'Predicted Price (Lakhs)': price_predictions
    })
    return df



# Streamlit app with tabs
st.title("Bangalore Home Price Prediction")


st.header("Input the Property Details")

# Input fields inside the first tab
location = st.selectbox("Select Location", data_columns[3:])  # Assuming first 3 columns are sqft, bath, bhk
sqft = st.number_input("Total Square Feet", min_value=500)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)

# Estimate Price button
# Estimate Price button
if st.button("Estimate Price"):
    price = predict_price(location, sqft, bath, bhk)
    
    if price < 0:
        st.error("No such property exists, please try higher values.")
    if price < 5 and price > 0:
        st.error(f"The estimated price for the home is ₹5.00 Lakhs")        
    else:
        st.success(f"The estimated price for the home is ₹{price:.2f} Lakhs") 


st.header(f"Price to Square-Feet Plot ")
df_plot = create_plot_data(location, bath, bhk)

# Create a plotly scatter plot for linear regression
fig = px.line(df_plot, x='Square Feet', y='Predicted Price (Lakhs)',
              labels={'Square Feet': 'Square Feet', 'Predicted Price (Lakhs)': 'Price (Lakhs)'},
              range_y=[5, df_plot['Predicted Price (Lakhs)'].max()])  # Set minimum y-axis value to 5 Lakhs

# Display the plot
st.plotly_chart(fig)
