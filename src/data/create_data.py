import numpy as np
import pandas as pd

from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import pandas as pd
import chromadb


# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma")
client.delete_collection("smart_connect_collection_v2")
# Create or get the collection
collection = client.create_collection("smart_connect_collection_v2", get_or_create=True)

# Example documents to add to the collection
documents = [
    {
        "id": "sales_doc0",
        "text": "Analyze the sales data for the last quarter using line charts",
        "metadata": {"type": "sales", "chart_type": "line"},
        "smart_connects": ["sales1"]
    },
    {
        "id": "sales_doc2",
        "text": "Review the sales data for the last quarter only in line charts",
        "metadata": {"type": "sales"},
        "smart_connects": ["sales2"]
    },
    {
        "id": "sales_doc3",
        "text": "Analyze data for sales using mix of charts",
        "metadata": {"type": "sales"},
        "smart_connects": ["sales1"]
    },
    {
        "id": "sales_doc4",
        "text": "Which products are selling the most",
        "metadata": {"type": "sales"},
        "smart_connects": ["sales1"]
    },
    {
        "id": "sales_doc5",
        "text": "What is the total revenue for the last quarter",
        "metadata": {"type": "sales"},
        "smart_connects": ["sales1"]
    },

    {
        "id": "feedback_doc1",
        "text": "how many customers are satisfied with the product",
        "metadata": {"type": "feedback"},
        "smart_connects": ["feedback1"]
    },
    {
        "id": "feedback_doc2",
        "text": "what customers are saying about the product",
        "metadata": {"type": "feedback"},
        "smart_connects": ["feedback1"]
    },
    {
        "id": "feedback_doc3",
        "text": "Review customer feedback from March",
        "metadata": {"type": "feedback"},
        "smart_connects": ["feedback1"]
    },

    # for employee data
    {
        "id": "employee_doc1",
        "text": "How many employees are there in the company",
        "metadata": {"type": "employee"},
        "smart_connects": ["employee1"]
    },
    {
        "id": "employee_doc2",
        "text": "What is the average age of employees",
        "metadata": {"type": "employee"},
        "smart_connects": ["employee1"]
    },
    {
        "id": "employee_doc3",
        "text": "What is the average salary of employees",
        "metadata": {"type": "employee"},
        "smart_connects": ["employee1"]
    },
    {
        "id": "employee_doc4",
        "text": "How many employees are there in the company",
        "metadata": {"type": "employee"},
        "smart_connects": ["employee1"]
    },
    {
        "id": "employee_doc5",
        "text": "What is the average age of employees",
        "metadata": {"type": "employee"},
        "smart_connects": ["employee1"]
    },
    {
        "id": "employee_doc6",
        "text": "What is the average salary of employees",
        "metadata": {"type": "employee"},
        "smart_connects": ["employee1"]
    },
    {
        "id": "feedback_doc4",
        "text": "dogs and cats are anmimal",
        "metadata": {"type": "employee"},
        "smart_connects": ["employee1"]
    },

]

document_map = {doc["id"]: doc for doc in documents}

# Add documents to the collection
collection.add(
    documents=[doc["text"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents],
    ids=[doc["id"] for doc in documents]
)
