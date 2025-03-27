from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from datetime import datetime
import pandas as pd


# Load the trained model
with open('prophet_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the input model using Pydantic
class OrderRequest(BaseModel):
    order_date: str  # Expected input is a string representing the date

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pizza Store Forecasting API!"}

# Define the forecasting endpoint
@app.post("/predict/")
def predict_sales(order: OrderRequest):
    # Convert the 'order_date' to a datetime object
    order_date = datetime.strptime(order.order_date, "%Y-%m-%d")

    # Prepare the input for the model (Prophet requires a DataFrame with 'ds' column)
    input_data = pd.DataFrame({'ds': [order_date]})

    # Make a prediction using the model
    forecast = model.predict(input_data)

    # Get the predicted sales (yhat)
    predicted_sales = forecast['yhat'].iloc[0]  # Get the prediction for the single date

    return {"predicted_sales": predicted_sales}
