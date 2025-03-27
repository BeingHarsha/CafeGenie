document.getElementById('orderForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const orderDate = document.getElementById('order_date').value;
    
    const response = await fetch('http://127.0.0.1:8000/predict/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ order_date: orderDate }),
    });
    
    const data = await response.json();
    
    const forecastResult = document.getElementById('forecastResult');
    
    if (data.predicted_sales) {
      forecastResult.innerHTML = `<h3>Predicted Sales: ${data.predicted_sales.join(", ")}</h3>`;
    } else {
      forecastResult.innerHTML = `<h3>Sorry, there was an error predicting the sales.</h3>`;
    }
  });
  