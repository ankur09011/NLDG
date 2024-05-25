""" Creates a sentiment analysis App using Taipy"""
import random

import taipy.gui
import json
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from pprint import pprint
import numpy as np
import pandas as pd
from taipy.gui import Gui, notify, builder as tgb
from constants.generic_constants import PAGE_1 as page, PAGE_RAW_DB
from data_models.sales_model import (
    SalesDataModel, Product, Customer, SalesTransaction, DATA_POINT_MAPPING_V2)

from data_models.smart_connects import smart_bridge
import chromadb

from data.create_data import documents, document_map

gen_text_label = ""
gen_bar_chart_df = pd.DataFrame()
gen_line_chart_df = pd.DataFrame()
gen_table_df = pd.DataFrame()

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

dataframe = pd.DataFrame()

dataframe1 = pd.DataFrame(
    {
        "Text": [""],
        "Score Pos": [0.33],
        "Score Neu": [0.33],
        "Score Neg": [0.33],
        "Overall": [0],
    }
)

dataframe2 = dataframe1.copy()

# raw query DF
raw_query_df = pd.read_csv('data/sample_sales_data.csv')

raw_text = pd.DataFrame(documents)

# convert metadata and smart_connects column to string from dict and list
raw_text['metadata'] = raw_text['metadata'].apply(lambda x: json.dumps(x))
raw_text['smart_connects'] = raw_text['smart_connects'].apply(lambda x: json.dumps(x))

print(raw_text)


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
    print("--------local_callback--------")
    from taipy.gui.state import State
    pprint(state)
    pprint(state.get_gui())

    try:
        import inspect
        print(inspect.getmembers(state))
        for i in inspect.getmembers(state):
            print(i)
    except Exception as e:
        print(e)

    # get all attributes of the state

    notify(state, "Info", f"The text is: {state.text}", True)
    temp = state.dataframe1.copy()
    scores = analyze_text(state.text)
    temp.loc[len(temp)] = scores
    state.dataframe1 = temp
    state.GLOBAL_QUERY = state.text

    # process smart bridge
    r = smart_bridge.run_pipeline_for_query(state.text)
    pprint(r)
    result_df = r['result_df']
    state.dataframe = result_df
    state.refresh('dataframe')

    # calling partial
    show_partial(state)


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
                              y=each_item['y'],
                              color='red')
        dynamic_components.append(bar_chart)

    elif each_item['type'] == 'line_chart':
        line_df = pd.DataFrame(sales_pipeline_results['data_points'][each_item['data_point']])
        line_chart = tgb.chart("{line_df}", type="line",
                               x=each_item['x'],
                               y=each_item['y'],
                               color='green')
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


new_partial_page = tgb.Page()
new_partial_page.add(tgb.text("## Partial Page"))

gui = Gui()
new_partial = gui.add_partial(new_partial_page)

my_text = "111"


def show_partial(state):
    notify(state, "Info", f"Showing partial", True)

    user_text = state.text

    r = smart_bridge.run_pipeline_for_query(user_text)
    print(r)
    dash_config = r.get('dash_config', {})
    _pipeline_results = r.get('sales_pipeline_results', {})
    _dynamic_components = []

    for data_point in dash_config.data_points:
        data_point_config = DATA_POINT_MAPPING_V2.get(data_point, {})
        print(data_point_config)
        print(_pipeline_results)

        dp_title = data_point_config.get('title', data_point.capitalize())
        possible_charts = data_point_config.get('visual_element', [])
        x_axis = data_point_config.get('x', 'x')
        y_axis = data_point_config.get('y', 'y')

        print(dp_title, possible_charts, x_axis, y_axis)
        if not possible_charts:
            continue
        chosen_charts = random.sample(possible_charts,
                                      min(dash_config.num_charts, len(possible_charts)))
        for chart_type in chosen_charts:
            color = random.choice(dash_config.color_palette)
            if chart_type == 'label':
                gen_text_label = f"{dp_title}: {_pipeline_results['data_points'][data_point]}"
                state.gen_text_label = gen_text_label
                text_component = tgb.text("## {gen_text_label}", mode='md')
                _dynamic_components.append(text_component)

            elif chart_type == 'bar_chart':
                gen_bar_chart_df = pd.DataFrame(_pipeline_results['data_points'][data_point])
                state.gen_bar_chart_df = gen_bar_chart_df
                bar_chart = tgb.chart("{gen_bar_chart_df}",
                                      type="bar",
                                      x=x_axis, y=y_axis, color=color,
                                      rebuild=True)
                _dynamic_components.append(bar_chart)

            elif chart_type == 'line_chart':
                gen_line_chart_df = pd.DataFrame(_pipeline_results['data_points'][data_point])
                state.gen_line_chart_df = gen_line_chart_df
                line_chart = tgb.chart("{gen_line_chart_df}",
                                       type="line",
                                       x=x_axis, y=y_axis, color=color,
                                       rebuild=True)
                _dynamic_components.append(line_chart)

            elif chart_type == 'table':
                gen_table_df = pd.DataFrame(_pipeline_results['data_points'][data_point])
                state.gen_table_df = gen_table_df
                table = tgb.table("{gen_table_df}", page_size=50, rebuild=True)
                _dynamic_components.append(table)

    _dynamic_page = tgb.Page()
    _dynamic_page.add(*_dynamic_components)
    new_partial.update_content(state, _dynamic_page)
    state.show_dialog = True



# test page
test_page = tgb.Page()
show_part = True
with test_page:
    tgb.text("## Natural Language Dashboard Generation", mode="md")

    tgb.layout(columns="1")
    tgb.text("**Your Query:** {text}", mode="md")

    tgb.text("Enter query for smart generation:")
    tgb.input("{text}", on_change="on_change")
    tgb.button("Analyze", on_action="local_callback")

    tgb.layout(columns="1")
    tgb.expandable("Query Table with matches ")
    tgb.table("{dataframe}", width="100%", number_format="%.2f", rebuild=True)
    tgb.layout(columns="1")
    tgb.table("{dataframe1}", width="100%", number_format="%.2f", rebuild=True)

    tgb.text("last query softmax score")
    tgb.layout(columns="1 1 1")
    tgb.text("Positive")
    tgb.text("{np.mean(dataframe1['Score Pos'])}", format="%.2f", raw=True)
    tgb.text("Neutral")
    tgb.text("{np.mean(dataframe1['Score Neu'])}", format="%.2f", raw=True)
    tgb.text("Negative")
    tgb.text("{np.mean(dataframe1['Score Neg'])}", format="%.2f", raw=True)

    tgb.text("This is a {my_text}")
    tgb.button("Show Partial", on_action="show_partial")
    tgb.part("{show_part}", partial="{new_partial}")

dialog_comp = "<|{show_dialog}|dialog|partial={new_partial}|width=800px|on_action=dialog_action|>"

pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "line": page + dialog_comp,
    "text": page_file,
    "dynamic": dynamic_page,
    "raw_data": PAGE_RAW_DB,
    "test_page": test_page
}

gui.add_pages(pages)

gui.run(title="NLDG", port=5001, debug=True, use_reloader=True)
