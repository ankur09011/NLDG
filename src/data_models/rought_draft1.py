from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import pandas as pd
import chromadb


class BaseDataModel(BaseModel):
    result_data: Dict[str, Any] = {}

    def preprocess(self):
        raise NotImplementedError("Preprocess method must be implemented by the subclass")

    def validate_data(self):
        raise NotImplementedError("Validate method must be implemented by the subclass")

    def benchmark(self):
        raise NotImplementedError("Benchmark method must be implemented by the subclass")

    def calculate_data_points(self):
        raise NotImplementedError("Calculate data points method must be implemented by the subclass")

    def run_pipeline(self, config: Optional[Dict[str, Any]] = None):
        self.preprocess()
        self.validate_data()
        self.benchmark()
        self.calculate_data_points()
        return self.result_data


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
    transaction_id: int
    product_id: int
    customer_id: int
    quantity: int
    total_amount: float
    date: str

    @field_validator('date')
    def validate_date(cls, v):
        return pd.to_datetime(v)


class SalesDataModel(BaseDataModel):
    products: List[Product]
    customers: List[Customer]
    transactions: List[SalesTransaction]

    def preprocess(self):
        print("Preprocessing sales data...")
        for transaction in self.transactions:
            transaction.date = pd.to_datetime(transaction.date)

    def validate_data(self):
        print("Validating sales data...")
        for transaction in self.transactions:
            if transaction.quantity <= 0:
                self.result_data['validation'] = 'Failed: Quantity must be positive'
                raise ValueError("Quantity must be positive")
            if transaction.total_amount <= 0:
                self.result_data['validation'] = 'Failed: Total amount must be positive'
                raise ValueError("Total amount must be positive")
        self.result_data['validation'] = 'Passed'

    def benchmark(self):
        print("Benchmarking sales data...")
        total_amounts = [t.total_amount for t in self.transactions]
        avg_transaction_value = sum(total_amounts) / len(total_amounts)
        self.result_data['benchmark'] = {'average_transaction_value': avg_transaction_value}

    def calculate_data_points(self):
        print("Calculating data points...")
        df = pd.DataFrame([t.dict() for t in self.transactions])
        total_sales = df['total_amount'].sum()
        top_products = df.groupby('product_id')['total_amount'].sum().nlargest(5).reset_index()
        avg_sales_per_product = df.groupby('product_id')['total_amount'].mean().reset_index()
        sales_over_time = df.groupby(df['date'].dt.to_period('M'))['total_amount'].sum().reset_index()

        self.result_data['data_points'] = {
            'total_sales': total_sales,
            'top_products': top_products.to_dict(orient='records'),
            'avg_sales_per_product': avg_sales_per_product.to_dict(orient='records'),
            'sales_over_time': sales_over_time.to_dict(orient='records')
        }


# Example usage of SalesDataModel
sales_data = SalesDataModel(
    products=[Product(product_id=1, name="Product A", category="Category 1", price=100.0)],
    customers=[Customer(customer_id=1, name="Customer A", location="Location 1", age=30)],
    transactions=[
        SalesTransaction(transaction_id=1, product_id=1, customer_id=1, quantity=2, total_amount=200.0,
                         date="2024-05-01"),
        SalesTransaction(transaction_id=2, product_id=1, customer_id=1, quantity=3, total_amount=300.0,
                         date="2024-05-02")
    ]
)


class BaseSmartConnect(BaseModel):
    id: str
    name: str

    def get_data_model(self):
        raise NotImplementedError("Each SmartConnect must implement the get_data_model method")


class SalesSmartConnect(BaseSmartConnect):
    data_model: SalesDataModel

    def get_data_model(self):
        return self.data_model


# Example usage
sales_smart_connect = SalesSmartConnect(
    id="sales1",
    name="Sales Data Connect",
    data_model=sales_data
)


class BaseDocument(BaseModel):
    text: str
    metadata: Dict[str, Any]
    smart_connects: List[str]  # References to SmartConnect objects by their IDs


# Initialize ChromaDB client
client = chromadb.PersistentClient()

# Create or get the collection
collection = client.create_collection("smart_connect_collection", get_or_create=True)

# Example documents to add to the collection
documents = [
    {
        "text": "Analyze the sales data for the last quarter",
        "metadata": {"type": "sales"},
        "smart_connects": ["sales1"]
    },
    {
        "text": "Review customer feedback from January",
        "metadata": {"type": "feedback"},
        "smart_connects": ["feedback1"]
    }
]

# Convert documents to BaseDocument instances
base_documents = [BaseDocument(**doc) for doc in documents]

# Add documents to the collection
collection.add(
    documents=[doc.text for doc in base_documents],
    metadatas=[doc.metadata for doc in base_documents],
    ids=[doc.metadata['type'] + "_doc" + str(i) for i, doc in enumerate(base_documents)]
)

# Query the collection
results = collection.query(
    query_texts=["What are the top  products?"],
    n_results=2
)
print(results)

# Process query results
for result in results['documents']:
    print(result)
