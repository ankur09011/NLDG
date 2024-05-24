from pydantic import BaseModel, field_validator, Field
from typing import List, Optional, Dict, Any
import pandas as pd
from jinja2 import Template

from .base_models import BaseDataModel


class Product(BaseModel):
    product_id: int
    name: str
    category: str
    price: float


class Customer(BaseModel):
    customer_id: int
    name: str
    location: str
    age: int


class SalesTransaction(BaseModel):
    invoice_id: str = Field(alias="Invoice ID")
    branch: str = Field(alias="Branch")
    city: str = Field(alias="City")
    customer_type: str = Field(alias="Customer_type")
    gender: str = Field(alias="Gender")
    product_line: str = Field(alias="Product line")
    unit_price: float = Field(alias="Unit price")
    quantity: int = Field(alias="Quantity")
    tax: float = Field(alias="Tax 5%")
    total: float = Field(alias="Total")
    date: str = Field(alias="Date")
    time: str = Field(alias="Time")
    payment: str = Field(alias="Payment")
    cogs: float = Field(alias="cogs")
    gross_margin_percentage: float = Field(alias="gross margin percentage")
    gross_income: float = Field(alias="gross income")
    rating: float = Field(alias="Rating")


class SalesDataModel(BaseDataModel):
    products: List[Product]
    customers: List[Customer]
    transactions: List[SalesTransaction]

    def preprocess(self):
        print("Preprocessing sales data...")
        # Example: Ensure all dates are converted to datetime objects
        for transaction in self.transactions:
            transaction.date = pd.to_datetime(transaction.date)

    def validate_data(self):
        print("Validating sales data...")
        # Example: Ensure no negative quantities or total amounts
        for transaction in self.transactions:
            if transaction.quantity <= 0:
                self.result_data['validation'] = 'Failed: Quantity must be positive'
                raise ValueError("Quantity must be positive")
            if transaction.total <= 0:
                self.result_data['validation'] = 'Failed: Total amount must be positive'
                raise ValueError("Total amount must be positive")
        self.result_data['validation'] = 'Passed'

    def benchmark(self):
        print("Benchmarking sales data...")
        # Example: Calculate average transaction value
        total_amounts = [t.total for t in self.transactions]
        avg_transaction_value = sum(total_amounts) / len(total_amounts)
        self.result_data['benchmark'] = {'average_transaction_value': avg_transaction_value}

    def calculate_data_points(self):
        print("Calculating data points...")
        df = pd.DataFrame([t.dict() for t in self.transactions])
        total_sales = df['total'].sum()
        top_products = df.groupby('product_line')['total'].sum().nlargest(5).reset_index()
        avg_sales_per_product = df.groupby('product_line')['total'].mean().reset_index()
        sales_over_time = df.groupby(df['date'].dt.to_period('M'))['total'].sum().reset_index()

        # convert period to string
        sales_over_time['date'] = sales_over_time['date'].astype(str)

        self.result_data['data_points'] = {
            'total_sales': total_sales,
            'top_products': top_products.to_dict(orient='records'),
            'avg_sales_per_product': avg_sales_per_product.to_dict(orient='records'),
            'sales_over_time': sales_over_time.to_dict(orient='records')
        }


class BaseSmartConnect(BaseModel):
    id: str
    name: str

    def get_data_model(self):
        raise NotImplementedError("Each SmartConnect must implement the get_data_model method")


class SalesSmartConnect(BaseSmartConnect):
    data_model: SalesDataModel
    dash_config: Dict[str, Any]

    def get_data_model(self):
        return self.data_model


class Feedback(BaseModel):
    feedback_id: int
    customer_id: int
    rating: float
    comments: Optional[str]
    date: str

    @field_validator('date')
    def validate_date(cls, v):
        return pd.to_datetime(v)


class CustomerFeedbackModel(BaseDataModel):
    feedbacks: List[Feedback]
    transactions: List[SalesTransaction]

    def preprocess(self):
        print("Preprocessing customer feedback data...")
        # Example: Convert date strings to datetime objects
        for feedback in self.feedbacks:
            feedback.date = pd.to_datetime(feedback.date)

    def validate_data(self):
        print("Validating customer feedback data...")
        # Example: Ensure ratings are between 1 and 5
        for feedback in self.feedbacks:
            if feedback.rating < 1 or feedback.rating > 5:
                self.result_data['validation'] = 'Failed: Rating must be between 1 and 5'
                raise ValueError("Rating must be between 1 and 5")
        self.result_data['validation'] = 'Passed'

    def benchmark(self):
        print("Benchmarking customer feedback data...")
        # Example: Calculate average rating
        ratings = [f.rating for f in self.feedbacks]
        avg_rating = sum(ratings) / len(ratings)
        self.result_data['benchmark'] = {'average_rating': avg_rating}

    def calculate_data_points(self):
        print("Calculating data points...")
        df = pd.DataFrame([f.dict() for f in self.feedbacks])
        avg_rating = df['rating'].mean()
        feedback_count = df.shape[0]
        ratings_distribution = df['rating'].value_counts().reset_index()

        self.result_data['data_points'] = {
            'avg_rating': avg_rating,
            'feedback_count': feedback_count,
            'ratings_distribution': ratings_distribution.to_dict(orient='records')
        }

# Example usage of CustomerFeedbackModel
# feedback_data = CustomerFeedbackModel(
#     feedbacks=[
#         Feedback(feedback_id=1, customer_id=1, rating=4.5, comments="Great service", date="2024-05-01"),
#         Feedback(feedback_id=2, customer_id=1, rating=3.5, comments="Good service", date="2024-05-02")
#     ],
#
# )
#
# # Customer feedback data pipeline
# feedback_pipeline_results = feedback_data.run_pipeline()
# print(feedback_pipeline_results)
# {'validation': 'Passed', 'benchmark': {'average_transaction_value': 250.0}, 'data_points': {'total_sales': 500.0, 'top_products': [{'product_id': 1, 'total_amount': 500.0}], 'avg_sales_per_product': [{'product_id': 1, 'total_amount': 250.0}], 'sales_over_time': [{'date': Period('2024-05', 'M'), 'total_amount': 500.0}]}}

# get total_sales from sales_pipeline_results data_points
# total_sales = sales_pipeline_results['data_points']['total_sales']
