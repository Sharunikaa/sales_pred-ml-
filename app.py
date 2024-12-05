from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import plotly.io as pio

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load and preprocess the dataset
df = pd.read_csv('agrisales.csv', delimiter=',', encoding='utf-8')
df.columns = df.columns.str.strip()
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], format='%Y-%m-%d')
df['Month'] = df['Purchase_Date'].dt.month
df['Day'] = df['Purchase_Date'].dt.day
df['Day_of_Week'] = df['Purchase_Date'].dt.dayofweek

# Feature selection and model training
X = df[['Quantity_Purchased', 'Price_per_Unit', 'Seasonality_Indicator', 'Promotional_Activity', 'Month', 'Day', 'Day_of_Week']]
y = df['Total_Sales']
X = pd.get_dummies(X, columns=['Seasonality_Indicator', 'Promotional_Activity'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
df['Predicted_Sales'] = model.predict(X)

# Dynamic Pricing Function
threshold = 15
def dynamic_pricing(predicted_sales, current_price):
    if predicted_sales > threshold:
        return current_price * 1.1
    else:
        return current_price * 0.9

df['Adjusted_Price'] = df.apply(lambda row: dynamic_pricing(row['Predicted_Sales'], row['Price_per_Unit']), axis=1)

# Customer Segmentation
customer_data = df.groupby(['Customer_ID', 'Product_Type'])['Quantity_Purchased'].sum().unstack().fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Segment'] = kmeans.fit_predict(customer_data)
def map_segment(row):
    if row['Rice'] > row['Wheat'] and row['Rice'] > row['Corn']:
        return 1
    elif row['Wheat'] > row['Rice'] and row['Wheat'] > row['Corn']:
        return 2
    else:
        return 3
customer_data['Segment'] = customer_data.apply(map_segment, axis=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicted_sales_graph')
def predicted_sales_graph():
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df['Purchase_Date'], y=df['Total_Sales'], mode='markers', name='Actual Sales', marker=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Purchase_Date'], y=df['Predicted_Sales'], mode='lines', name='Predicted Sales', line=dict(dash='dash', color='red')), row=1, col=1)
    fig.update_layout(title='Actual vs Predicted Sales', xaxis_title='Date', yaxis_title='Sales')
    graph_html = pio.to_html(fig, full_html=False)
    return render_template('predicted_sales_graph.html', graph_html=graph_html)

@app.route('/predict_future_sales', methods=['GET', 'POST'])
def predict_future_sales():
    if request.method == 'POST':
        quantity_purchased = int(request.form['quantity_purchased'])
        price_per_unit = float(request.form['price_per_unit'])
        seasonality_indicator = request.form['seasonality_indicator']
        promotional_activity = request.form['promotional_activity']
        purchase_date = request.form['purchase_date']
        predicted_sales, adjusted_price, suggestion = predict_sales(quantity_purchased, price_per_unit, seasonality_indicator, promotional_activity, purchase_date)
        return render_template('predict_future_sales.html', predicted_sales=predicted_sales, adjusted_price=adjusted_price, suggestion=suggestion)
    return render_template('predict_future_sales.html')

@app.route('/customer_segmentation')
def customer_segmentation():
    fig = go.Figure()
    for i in range(3):
        segment_data = customer_data[customer_data['Segment'] == i + 1]
        fig.add_trace(go.Scatter3d(x=segment_data['Rice'], y=segment_data['Wheat'], z=segment_data['Corn'], mode='markers', marker=dict(color=px.colors.qualitative.Plotly[i], size=5), name=f'Segment {i + 1}'))
    fig.update_layout(scene=dict(xaxis_title='Rice', yaxis_title='Wheat', zaxis_title='Corn'), title='Customer Segmentation', legend=dict(title='Segment'))
    graph_html = pio.to_html(fig, full_html=False)
    return render_template('customer_segmentation.html', graph_html=graph_html)

def predict_sales(quantity_purchased, price_per_unit, seasonality_indicator, promotional_activity, purchase_date):
    purchase_date = pd.to_datetime(purchase_date, format='%Y-%m-%d')
    month = purchase_date.month
    day = purchase_date.day
    day_of_week = purchase_date.dayofweek
    new_data = pd.DataFrame({
        'Quantity_Purchased': [quantity_purchased],
        'Price_per_Unit': [price_per_unit],
        'Seasonality_Indicator': [seasonality_indicator],
        'Promotional_Activity': [promotional_activity],
        'Month': [month],
        'Day': [day],
        'Day_of_Week': [day_of_week]
    })
    new_data = pd.get_dummies(new_data, columns=['Seasonality_Indicator', 'Promotional_Activity'], drop_first=True)
    for col in X.columns:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[X.columns]
    predicted_sales = model.predict(new_data)[0]
    adjusted_price = dynamic_pricing(predicted_sales, price_per_unit)
    if adjusted_price > price_per_unit:
        suggestion = f"Increase the price by {((adjusted_price - price_per_unit) / price_per_unit) * 100:.2f}%"
    elif adjusted_price < price_per_unit:
        suggestion = f"Decrease the price by {((price_per_unit - adjusted_price) / price_per_unit) * 100:.2f}%"
    else:
        suggestion = "Maintain the current price"
    return predicted_sales, adjusted_price, suggestion

if __name__ == '__main__':
    app.run(debug=True)
