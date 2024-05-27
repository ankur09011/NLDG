# Building a Smart Module for Automatic Analytics Dashboard Creation

## Inspiration and Problem Statement

The idea for creating a smart module for automatic analytics dashboard generation emerged during a casual dinner conversation with my senior. We discussed the challenges of creating dynamic, insightful dashboards that could cater to various business needs without manual intervention.

## Note: This project is a work in progress. The code snippets and explanations are based on the initial design and implementation. The final version may differ significantly.

SNAPSHOTS:
raw dashboard
![raw_image_dashboard.png](docs%2Fraw_image_dashboard.png)

result df
![raw_image_for_search.png](docs%2Fraw_image_for_search.png)

DEMO VIDEO: 
![24.05.2024 23_40.gif](docs%2F24.05.2024%2023_40.gif)

### Goal

To develop a system that:
- Understands user queries.
- Identifies relevant data models.
- Extract insights from the data based on the nature of the query.
- Runs data processing pipelines to generate insightful dashboards.

## Design Choices and Evolution

### 1. Base Data Model

#### Design Choice

We needed a standardized way to handle data processing across different use cases. This led to the creation of a `BaseDataModel` class, which would define the basic structure for preprocessing, validation, benchmarking, and calculating data points.

#### Implementation

- **Preprocessing**: Convert data into a suitable format for analysis.
- **Validation**: Ensure data integrity and consistency.
- **Benchmarking**: Evaluate the data against certain metrics.
- **Data Points Calculation**: Extract key metrics and insights from the data.

```python
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

```
### 2. Specific Data Models
#### Design Choice

We needed specific implementations of BaseDataModel for different analytics scenarios, such as sales data and customer feedback.

#### Implementation

Each specific data model would implement the methods defined in BaseDataModel to handle the unique aspects of its data.

```
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
            if transaction.total <= 0:
                self.result_data['validation'] = 'Failed: Total amount must be positive'
                raise ValueError("Total amount must be positive")
        self.result_data['validation'] = 'Passed'

    def benchmark(self):
        print("Benchmarking sales data...")
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

        self.result_data['data_points'] = {
            'total_sales': total_sales,
            'top_products': top_products.to_dict(orient='records'),
            'avg_sales_per_product': avg_sales_per_product.to_dict(orient='records'),
            'sales_over_time': sales_over_time.to_dict(orient='records')
        }
```

### 3. Semantic Search Integration
#### Design Choice

To dynamically match user queries with the appropriate data models, we integrated ChromaDB for managing documents and performing semantic searches.

#### Implementation

Document Management: Store documents pointing to SmartConnect objects.
Semantic Search: Match user queries with documents to find relevant data models.
### 4. SmartBridge Class
#### Design Choice

The SmartBridge class handles query results from ChromaDB and runs the appropriate data pipelines.

#### Implementation

Query Processing: Map query results to SmartConnect objects and execute their pipelines.
```
class SmartBridge(BaseModel):
    query_results: List[Dict[str, Any]]
    smart_connects: Dict[str, BaseSmartConnect]

    def run_pipeline_for_query(self, query_text: str):
        results = collection.query(query_texts=[query_text], n_results=1)
        for i, result_list in enumerate(results['documents']):
            for j, result in enumerate(result_list):
                document = BaseDocument(
                    text=result,
                    metadata=results['metadatas'][i][j],
                    smart_connects=results['metadatas'][i][j].get('smart_connects', [])
                )
                for connect_id in document.smart_connects:
                    if connect_id in self.smart_connects:
                        smart_connect = self.smart_connects[connect_id]
                        data_model = smart_connect.get_data_model()
                        pipeline_results = data_model.run_pipeline()
                        print(f"Pipeline results for {smart_connect.name}: {pipeline_results}")

```
### 5. Dynamic Page Creation Using Taipy
#### Design Choice

To create interactive and dynamic dashboards, we used Taipy's Page Builder API.

#### Implementation

Page Components: Use Taipy's API to create headings, selectors, charts, and tables.
Configuration-Based Design: Define configurations to dynamically generate visual components based on data points.
```
class PageBuilder:
    def __init__(self, config: dict):
        self.config = config

    def create_page(self):
        df = self.config['dataframe']
        unique_products = df['Product line'].unique()
        unique_customers = df['Customer_type'].unique()
        unique_transactions = df['Invoice ID'].unique()
        unique_cities = df['City'].unique()
        unique_branches = df['Branch'].unique()

        def on_button_action(state):
            notify(state, 'info', f'The text is: {state.text}')
            state.text = "Button Pressed"

        def on_change(state, var_name, var_value):
            if var_name == "text" and var_value == "Reset":
                state.text = ""
                return

        text = "Original text"
        generic_page = tgb.Page()

        # Headings
        main_heading = tgb.text("# Getting started with Taipy GUI", mode="md")
        sub_heading = tgb.text("My text: {text}")

        # Inputs
        text_input = tgb.input("{text}")

        # Button
        run_button = tgb.button("Run local", on_action=on_button_action)

        # Add a selector for the city
        sel_city = unique_cities[0]
        city_selector = tgb.selector("{sel_city}", lov="{unique_cities}", dropdown=True,)

        # Create the Plotly figure object
        figure = go.Figure()
        figure.add_trace(go.Violin(name="Normal", y=np.random.normal(loc=0, scale=1, size=1000), box_visible=True, meanline_visible=True))
        figure.add_trace(go.Violin(name="Exponential", y=np.random.exponential(scale=1, size=1000), box_visible=True, meanline_visible=True))
        figure.add_trace(go.Violin(name="Uniform", y=np.random.uniform(low=0, high=1, size=1000), box_visible=True, meanline_visible=True))
        figure.update_layout(title="Different Probability Distributions")

        demo_chart = tgb.chart(figure="{figure}")

        list_to_display = [100/x for x in range(1, 100)]
        fig = go.Figure(data=go.Scatter(y=list_to_display))

        demo_chart_2 = tgb.chart(figure="{fig}")

        # Add table
        demo_table = tgb.table("{df}")

        # Add components to the page
        components = [main_heading, sub_heading, text_input, run_button, city_selector, demo_chart, demo_chart_2, demo_table]
        generic_page.add(*components)

        return generic_page

    def run(self):
        page_content = self.create_page()
        gui = Gui(page_content)
        gui.run()

# Sample Data and Configuration
df = pd.read_csv('/mnt/data/sample_sales_data.csv')
config = {'dataframe': df}
page_builder = PageBuilder(config=config)
page_builder.run()
```
### Conclusion
This POC demonstrates a 
quick scalable and flexible approach to creating automatic analytics dashboards based on user queries. By standardizing data models, leveraging semantic search, and dynamically connecting to appropriate data models, we can generate insightful dashboards efficiently.

### Future Work
* Extend Data Models: Add more specific data models for various analytics use cases. 
* Enhance Semantic Search: Improve query handling and matching algorithms. 
* Optimize Pipelines: Refine data processing pipelines for better performance and accuracy. 
* Documentation and Examples: Provide comprehensive documentation and examples to facilitate understanding and usage of the integrated semantic search and SmartConnect.


# Next Phase:

Discuss the next phase of the project, including planned features, improvements, and potential challenges.

Understanding is now we need think it as a virtual assistant 
which can answer the queries and provide the insights in the form of dashboard.

Now to do that one of the framework OpsDroid we can use


Design choices:
user ask a question we will run the query with following pipleine

1. Query Processing
2. NLP
3. RAG
4. OpsDroid
5. Voice Services

Now think it like mathcking a particular skill with the query and then running the pipeline
