""" Creates a sentiment analysis App using Taipy"""
import taipy.gui
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

import numpy as np
import pandas as pd
from taipy.gui import Gui, notify, builder as tgb
from constants.generic_constants import PAGE_1 as page
from data_models.sales_model import SalesDataModel, Product, Customer, SalesTransaction

import chromadb


# Initialize ChromaDB client
client = chromadb.PersistentClient(path="data/chroma")

# Create or get the collection
collection = client.create_collection("smart_connect_collection_v2", get_or_create=True)


text = "Original text"

GLOBAL_QUERY = "User Query"

MODEL = "sbcBI/sentiment_analysis_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
show_dialog = False

dataframe = pd.DataFrame(
    {
        "Text": [""],
        "Score Pos": [0.33],
        "Score Neu": [0.33],
        "Score Neg": [0.33],
        "Overall": [0],
    }
)

dataframe2 = dataframe.copy()


def analyze_text(input_text: str) -> dict:
    """
    Runs the sentiment analysis model on the text

    Args:
        - text (str): text to be analyzed

    Returns:
        - dict: dictionary with the scores
    """
    encoded_text = tokenizer(input_text, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return {
        "Text": input_text[:50],
        "Score Pos": scores[2],
        "Score Neu": scores[1],
        "Score Neg": scores[0],
        "Overall": scores[2] - scores[0],
    }


def local_callback(state) -> None:
    """
    Analyze the text and updates the dataframe

    Args:
        - state: state of the Taipy App
    """
    notify(state, "Info", f"The text is: {state.text}", True)
    temp = state.dataframe.copy()
    scores = analyze_text(state.text)
    temp.loc[len(temp)] = scores
    state.dataframe = temp
    state.GLOBAL_QUERY = state.text
    state.text = ""
    state.show_dialog = True

    # for the smart connect getting query

    results = collection.query(
        query_texts=[state.text],
        n_results=3  # Adjusted to match the example result structure
    )
    print(results)

path = ""
treatment = 0

page_file = """
<|{path}|file_selector|extensions=.txt|label=Upload .txt file|on_action=analyze_file|> <|{f'Downloading {treatment}%...'}|>

<br/>

<|Table|expandable|
<|{dataframe2}|table|width=100%|number_format=%.2f|>
|>

<br/>

<|{dataframe2}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|height=600px|>

"""


def on_change(state, var_name, var_value):
    print(f"Var name: {var_name}, Var value: {var_value} and State: {state}")
    if var_name == "text" and var_value == "Reset":
        state.text = ""
        return


def analyze_file(state) -> None:
    """
    Analyse the lines in a text file

    Args:
        - state: state of the Taipy App
    """
    state.dataframe2 = dataframe2
    state.treatment = 0
    with open(state.path, "r", encoding="utf-8") as f:
        data = f.read()
        print(data)

        file_list = list(data.split("\n"))

    for i, input_text in enumerate(file_list):
        state.treatment = int((i + 1) * 100 / len(file_list))
        temp = state.dataframe2.copy()
        scores = analyze_text(input_text)
        temp.loc[len(temp)] = scores
        state.dataframe2 = temp

    state.path = None

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

# Define a configuration for creating components
config = {
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

# Create components based on the configuration
# components = sales_data.create_components(config)
# print(components)


dynamic_components = []

for each_item in config['items']:
    print(each_item)
    if each_item['type'] == 'heading':
        text_component = tgb.text("## {GLOBAL_QUERY}")
        dynamic_components.append(text_component)

    elif each_item['type'] == 'bar_chart':
        print(each_item)
        bar_chart_df = pd.DataFrame(sales_pipeline_results['data_points'][each_item['data_point']])
        print(bar_chart_df)
        bar_chart = tgb.chart("{bar_chart_df}", type="bar",
                              x=each_item['x'],
                              y=each_item['y'])
        dynamic_components.append(bar_chart)

    elif each_item['type'] == 'line_chart':
        line_df = pd.DataFrame(sales_pipeline_results['data_points'][each_item['data_point']])
        line_chart = tgb.chart("{line_df}", type="line",
                               x=each_item['x'],
                               y=each_item['y'])
        dynamic_components.append(line_chart)

    elif each_item['type'] == 'table':
        table_df = pd.DataFrame(sales_pipeline_results['data_points'][each_item['data_point']])
        table = tgb.table("{table_df}", page_size=50)
        dynamic_components.append(table)

dynamic_page = tgb.Page()
dynamic_page.add(*dynamic_components)


def dialog_action(state, id, payload):
    with state as st:
        st.show_dialog = False


dialog_comp = "<|{show_dialog}|dialog|page=dynamic|width=800px|on_action=dialog_action|>"

pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "line": page + dialog_comp,
    "text": page_file,
    "dynamic": dynamic_page
}

Gui(pages=pages).run(title="Sentiment Analysis", port=5001, debug=True, use_reloader=True)
