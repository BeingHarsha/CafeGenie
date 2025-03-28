from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import pickle

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

# Define the request model
class ForecastRequest(BaseModel):
    start_date: str  # Expected format: YYYY-MM-DD

class OrderRequest(BaseModel):
    order_date: str  # Expected format: YYYY-MM-DD

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pizza Store Forecasting API!"}

@app.post("/predict/")
def predict_sales(request: ForecastRequest):
    try:
        # Convert input date to datetime object
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # Generate future dates for the next 7 days
        future_dates = [start_date + timedelta(days=i) for i in range(7)]
        
        # Prepare DataFrame for Prophet
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Predict sales
        forecast = model.predict(future_df)
        
        # Aggregate total sales for the next 7 days
        total_predicted_sales = forecast['yhat'].sum()
        
        return {"total_predicted_sales": round(total_predicted_sales, 2)}

    except ValueError as e:
        # Catch any date parsing errors
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/sales-forecast-week/")
def sales_forecast_week(order: OrderRequest):
    try:
        # Determine the start of the week (Monday)
        current_date = datetime.strptime(order.order_date, "%Y-%m-%d")
        start_of_week = current_date - timedelta(days=current_date.weekday())  # Start from Monday
        end_of_week = start_of_week + timedelta(days=6)  # End on Sunday

        # Create a range of dates for the week
        week_dates = pd.date_range(start=start_of_week, end=end_of_week)

        # Prepare input data for model (Prophet needs 'ds' column)
        input_data = pd.DataFrame({'ds': week_dates})

        # Predict sales for the entire week
        forecast = model.predict(input_data)

        # Aggregate the sales for the week
        aggregated_sales = forecast['yhat'].sum()

        return {"predicted_sales": aggregated_sales}

    except ValueError as e:
        # Catch any date parsing errors
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
