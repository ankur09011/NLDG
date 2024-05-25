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
from .sales_model import (SalesDataModel,
                          Product, Customer,
                          Feedback,
                          SalesTransaction,
                          CustomerFeedbackModel,
                          EmployeeDataModel,
                          DATA_MODEL_MAPPING,
                          DATA_POINT_MAPPING_V2
                          )

from .dashboard_generator import DashboardConfig, DashboardGenerator

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="data/chroma")

# Create or get the collection
collection = client.create_collection("smart_connect_collection_v2", get_or_create=True)


class BaseSmartConnect(BaseModel):
    id: str
    name: str
    dash_config: DashboardConfig
    data_pipeline: str
    data_model: BaseDataModel

    def get_data_model(self):
        return self.data_model


class SalesSmartConnect(BaseSmartConnect):
    data_model: Optional[SalesDataModel] = None


class FeedbackSmartConnect(BaseSmartConnect):
    data_model: Optional[CustomerFeedbackModel] = None


class EmployeeSmartConnect(BaseSmartConnect):
    data_model: Optional[EmployeeDataModel] = None


# dynamic page

# Example usage
sales_smart_connect = SalesSmartConnect(
    id="sales1",
    name="Sales Data Connect",
    dash_config=DashboardConfig(
        data_points=['total_sales', 'top_products', 'sales_over_time'],
        num_charts=5,
        color_palette=['blue', 'green', 'red', 'purple']
    ),
    data_pipeline='sales'
)

feedback_smart_connect = FeedbackSmartConnect(
    id="feedback1",
    name="Feedback Data Connect",
    dash_config=DashboardConfig(
        data_points=['feedback_count', 'avg_rating', 'ratings_distribution'],
        num_charts=5,
        color_palette=['blue', 'green', 'red', 'purple']
    ),
    data_pipeline='feedback'
)

employee_smart_connect = EmployeeSmartConnect(
    id="employee1",
    name="Employee Data Connect",
    dash_config=DashboardConfig(
        data_points=['total_sales', 'avg_incentive',
                     'gender_distribution', 'age_distribution'],
        num_charts=5,
        color_palette=['blue', 'green', 'red', 'purple']
    ),
    data_pipeline='employee'
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
                    smart_data_pipeline = smart_connect.data_pipeline
                    data_model = DATA_MODEL_MAPPING.get(smart_data_pipeline)
                    pipeline_results = data_model.run_pipeline()
                    print(f"------{smart_connect.data_pipeline} ----- ")
                    print(f"Pipeline results for {smart_connect.name}: {pipeline_results}")
                    pipeline_results_list.append(pipeline_results)
                else:
                    print(f"SmartConnect with ID {connect_id} not found")
            #
            # dashboard_generator = DashboardGenerator(config=smart_connect.dash_config,
            #                                          sales_pipeline_results=pipeline_results,
            #                                          data_point_mapping=DATA_POINT_MAPPING_V2)
            # dynamic_page = dashboard_generator.generate_components()

            return {"pipeline_results": pipeline_results_list,
                    "query_result": results,
                    "result_df": result_df,
                    "smart_connects": self.smart_connects,
                    "dash_config": smart_connect.dash_config,
                    "sales_pipeline_results": pipeline_results, }


# Example usage of SmartBridge and smart connects
smart_connects = {
    "sales1": sales_smart_connect,
    "sales2": sales_smart_connect,

    # feedback
    "feedback1": feedback_smart_connect,
    "feedback2": feedback_smart_connect,

    # employee
    "employee1": employee_smart_connect,

    # customer

}

smart_bridge = SmartBridge(smart_connects=smart_connects)
