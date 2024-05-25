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


class Employee(BaseModel):
    employee_id: int
    name: str
    gender: str
    age: int
    doj: str
    attendance_percentage: float
    incentive: float
    sales: float


class EmployeeDataModel(BaseDataModel):
    employees: List[Employee]
    result_data: Dict[str, Any] = Field(default_factory=dict)

    def preprocess(self):
        print("Preprocessing employee data...")
        for employee in self.employees:
            employee.doj = pd.to_datetime(employee.doj)

    def validate_data(self):
        print("Validating employee data...")
        for employee in self.employees:
            if not (0 <= employee.attendance_percentage <= 100):
                self.result_data['validation'] = 'Failed: Attendance percentage must be between 0 and 100'
                raise ValueError("Attendance percentage must be between 0 and 100")
            if employee.age <= 0:
                self.result_data['validation'] = 'Failed: Age must be positive'
                raise ValueError("Age must be positive")
            if employee.incentive < 0:
                self.result_data['validation'] = 'Failed: Incentive cannot be negative'
                raise ValueError("Incentive cannot be negative")
            if employee.sales < 0:
                self.result_data['validation'] = 'Failed: Sales cannot be negative'
                raise ValueError("Sales cannot be negative")
        self.result_data['validation'] = 'Passed'

    def benchmark(self):
        print("Benchmarking employee data...")
        sales_amounts = [e.sales for e in self.employees]
        avg_sales = sum(sales_amounts) / len(sales_amounts)
        self.result_data['benchmark'] = {'average_sales': avg_sales}

    def calculate_data_points(self):
        print("Calculating data points...")
        df = pd.DataFrame([e.dict() for e in self.employees])
        total_sales = df['sales'].sum()
        avg_incentive = df['incentive'].mean()
        gender_distribution = df['gender'].value_counts().reset_index()
        age_distribution = df['age'].value_counts().reset_index()

        self.result_data['data_points'] = {
            'total_sales': total_sales,
            'avg_incentive': avg_incentive,
            'gender_distribution': gender_distribution.to_dict(orient='records'),
            'age_distribution': age_distribution.to_dict(orient='records')
        }


# Example Usage
employees = [
    Employee(employee_id=1, name="John Doe", gender="Male", age=30, doj="2020-01-15", attendance_percentage=95.5,
             incentive=2000, sales=15000),
    Employee(employee_id=2, name="Jane Smith", gender="Female", age=28, doj="2019-07-23", attendance_percentage=98.2,
             incentive=2500, sales=20000),
    # Add more employees as needed
]

employee_data_model = EmployeeDataModel(employees=employees)
employee_data_model.preprocess()
employee_data_model.validate_data()
employee_data_model.benchmark()
employee_data_model.calculate_data_points()
print(employee_data_model.result_data)

# TODO: NOTE: remove below code to make it based on user

# transaction data
sales_df = pd.read_csv('data/sample_sales_data.csv')
employee_df = pd.read_csv('data/sample_employee_data.csv')
feedback_df = pd.read_csv('data/sample_feedback_data.csv')

products = [
    Product(product_id=1, name="Product A", category="Category 1", price=100.0)
]
customers = [
    Customer(customer_id=1, name="Customer A", location="Location 1", age=30)
]

transactions = [SalesTransaction(**row) for row in sales_df.to_dict(orient='records')]
feedbacks = [Feedback(**row) for row in feedback_df.to_dict(orient='records')]
employees = [Employee(**row) for row in employee_df.to_dict(orient='records')]

# different data
sales_data = SalesDataModel(
    products=products,
    customers=customers,
    transactions=transactions
)

feedback_data = CustomerFeedbackModel(
    feedbacks=feedbacks,
    transactions=transactions
)

employee_data = EmployeeDataModel(
    employees=employees
)

DATA_MODEL_MAPPING = {
    'sales': sales_data,
    'feedback': feedback_data,
    'employee': employee_data
}

DATA_POINT_MAPPING = {
    'total_sales': ['label'],
    'top_products': ['bar_chart', 'table'],
    'sales_over_time': ['bar_chart', 'line_chart'],
    'feedback_count': ['label'],
    'avg_rating': ['label'],
    'ratings_distribution': ['bar_chart', 'table', 'line_chart'],
    'avg_incentive': ['label'],
    'gender_distribution': ['bar_chart', 'table', 'line_chart'],
    'age_distribution': ['bar_chart', 'table', 'line_chart']
}

DATA_POINT_MAPPING_V2 = {
    'total_sales': {
        'visual_element': ['label'],
        'title': 'Total Sales',
        'x': 'total_sales'
    },
    'top_products': {
        'visual_element': ['bar_chart', 'table'],
        'title': 'Top Products by Sales',
        'x': 'product_line',
        'y': 'total'
    },
    'sales_over_time': {
        'visual_element': ['bar_chart', 'line_chart'],
        'title': 'Sales Over Time',
        'x': 'date',
        'y': 'total'
    },
    'feedback_count': {
        'visual_element': ['label'],
        'title': 'Feedback Count',
        'x': 'feedback_count'
    },
    'avg_rating': {
        'visual_element': ['label'],
        'title': 'Average Rating',
        'x': 'avg_rating'
    },
    'ratings_distribution': {
        'visual_element': ['bar_chart', 'table', 'line_chart'],
        'title': 'Ratings Distribution',
        'x': 'rating',
        'y': 'count'
    },
    'avg_incentive': {
        'visual_element': ['label'],
        'title': 'Average Incentive',
        'x': 'avg_incentive'
    },
    'gender_distribution': {
        'visual_element': ['bar_chart', 'table', 'line_chart'],
        'title': 'Gender Distribution',
        'x': 'gender',
        'y': 'count'
    },
    'age_distribution': {
        'visual_element': ['bar_chart', 'table', 'line_chart'],
        'title': 'Age Distribution',
        'x': 'age',
        'y': 'count'
    }
}
