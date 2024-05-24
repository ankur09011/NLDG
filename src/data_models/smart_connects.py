from pydantic import BaseModel, validator, Field
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from taipy.gui import builder as tgb
import numpy as np
import plotly.graph_objects as go
import chromadb
from data.create_data import documents, document_map
from .base_models import BaseDataModel, BaseDocument
from .sales_model import SalesDataModel, Product, Customer, SalesTransaction

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="data/chroma")

# Create or get the collection
collection = client.create_collection("smart_connect_collection_v2", get_or_create=True)


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


dashboard_sales = {
    'items': [
        {'type': 'heading', 'label': 'Total Sales', 'data_point': 'total_sales'},
        {'type': 'bar_chart', 'title': 'Top Products by Sales', 'x': 'product_line', 'y': 'total',
         'data_point': 'top_products'},
        # {'type': 'line_chart', 'title': 'Top Products by Sales', 'x': 'product_line', 'y': 'total',
        #          'data_point': 'top_products'},
        {'type': 'table', 'title': 'Top Products by Sales', 'x': 'product_line', 'y': 'total',
         'data_point': 'top_products'},
        {'type': 'line_chart', 'title': 'Sales Over Time',
         'x': 'date', 'y': 'total', 'data_point': 'sales_over_time'}

    ]
}

# dynamic page
df = pd.read_csv('data/sample_sales_data.csv')
print(df)

# Example usage of SalesDataModel
sales_data = SalesDataModel(
    products=[
        Product(product_id=1, name="Product A", category="Category 1", price=100.0)
    ],
    customers=[
        Customer(customer_id=1, name="Customer A", location="Location 1", age=30)
    ],
    transactions=[SalesTransaction(**row) for row in df.to_dict(orient='records')]
)

# Sales data pipeline
sales_pipeline_results = sales_data.run_pipeline()
print(sales_pipeline_results)
print(sales_pipeline_results)

# Example usage
sales_smart_connect = SalesSmartConnect(
    id="sales1",
    name="Sales Data Connect",
    data_model=sales_data,
    dash_config=dashboard_sales
)


class SmartBridge(BaseModel):
    smart_connects: Dict[str, BaseSmartConnect]

    def run_pipeline_for_query(self, query_text: str):
        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=5  # Adjusted to match the example result structure
        )
        print("SmartBridge.run_pipeline_for_query")
        print(results)

        # Flatten the lists to create columns
        _distances = results['distances'][0]
        _documents = results['documents'][0]
        _ids = results['ids'][0]
        _metadatas = [metadata['type'] for metadata in results['metadatas'][0]]

        # Create a DataFrame
        result_df = pd.DataFrame({
            'Distance': _distances,
            'Document': _documents,
            'ID': _ids,
            'Metadata Type': _metadatas
        })

        # Display the DataFrame
        print(result_df)

        pipeline_results_list = []

        # Process query results and run the appropriate pipeline
        for i, doc_id in enumerate(results['ids'][0]):
            print(f"Processing result {i + 1}:, doc_id={doc_id}")
            # Retrieve the full document details using the document ID
            document = next(doc for doc in documents if doc["id"] == doc_id)
            print(document)
            base_document = BaseDocument(
                text=document["text"],
                metadata=document["metadata"],
                smart_connects=document["smart_connects"])

            print(base_document)
            for connect_id in base_document.smart_connects:
                if connect_id in self.smart_connects:
                    smart_connect = self.smart_connects[connect_id]
                    data_model = smart_connect.get_data_model()
                    pipeline_results = data_model.run_pipeline()
                    print(f"Pipeline results for {smart_connect.name}: {pipeline_results}")
                    pipeline_results_list.append(pipeline_results)
                else:
                    print(f"SmartConnect with ID {connect_id} not found")

            return {"pipeline_results": pipeline_results_list,
                    "query_result": results,
                    "result_df": result_df}


# Example usage of SmartBridge
smart_connects = {
    "sales1": sales_smart_connect,
    "sales2": sales_smart_connect,
    # Add other SmartConnect objects here
    "feedback1": sales_smart_connect
}

smart_bridge = SmartBridge(smart_connects=smart_connects)
