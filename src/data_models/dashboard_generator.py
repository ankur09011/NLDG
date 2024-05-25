from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
import random
import pandas as pd
import taipy.gui.builder as tgb

gen_text_label = ""
gen_bar_chart_df = pd.DataFrame()
gen_line_chart_df = pd.DataFrame()
gen_table_df = pd.DataFrame()


class DashboardConfig(BaseModel):
    data_points: List[str]
    num_charts: int = 3
    color_palette: List[str] = Field(
        default_factory=lambda: ['blue', 'green', 'red', 'purple'])

    model_config = ConfigDict(extra='allow')


class DashboardGenerator:
    def __init__(self, config: DashboardConfig,
                 sales_pipeline_results: Dict[str, Any],
                 data_point_mapping: Dict):
        self.config = config
        self.sales_pipeline_results = sales_pipeline_results
        self.data_point_mapping = data_point_mapping
        self.dynamic_components = []

    def generate_components(self):
        for data_point in self.config.data_points:
            data_point_config = self.data_point_mapping.get(data_point, {})
            print(data_point_config)
            print(sales_pipeline_results)

            dp_title = data_point_config.get('title', data_point.capitalize())
            possible_charts = data_point_config.get('visual_element', [])
            x_axis = data_point_config.get('x', 'x')
            y_axis = data_point_config.get('y', 'y')

            print(dp_title, possible_charts, x_axis, y_axis)
            if not possible_charts:
                continue
            chosen_charts = random.sample(possible_charts, min(self.config.num_charts, len(possible_charts)))
            print(chosen_charts)

            for chart_type in chosen_charts:
                color = random.choice(self.config.color_palette)
                if chart_type == 'label':
                    gen_text_label = f"{dp_title}: {self.sales_pipeline_results['data_points'][data_point]}"
                    print(gen_text_label)
                    text_component = tgb.text("## {gen_text_label}", mode='md')
                    self.dynamic_components.append(text_component)

                elif chart_type == 'bar_chart':
                    gen_bar_chart_df = pd.DataFrame(self.sales_pipeline_results['data_points'][data_point])
                    print(gen_bar_chart_df)
                    bar_chart = tgb.chart("{gen_bar_chart_df}",
                                          type="bar",
                                          x=x_axis, y=y_axis, color=color)
                    self.dynamic_components.append(bar_chart)

                elif chart_type == 'line_chart':
                    gen_line_chart_df = pd.DataFrame(self.sales_pipeline_results['data_points'][data_point])
                    print(gen_line_chart_df)
                    line_chart = tgb.chart("{gen_line_chart_df}",
                                           type="line",
                                           x=x_axis, y=y_axis, color=color)
                    self.dynamic_components.append(line_chart)

                    # df = pd.DataFrame(self.sales_pipeline_results['data_points'][data_point])
                    # line_chart = tgb.chart(df, type="line", x='date', y='total', color=color)
                    # self.dynamic_components.append(line_chart)

                elif chart_type == 'table':
                    gen_table_df = pd.DataFrame(self.sales_pipeline_results['data_points'][data_point])
                    print(gen_table_df)
                    table = tgb.table("{gen_table_df}", page_size=50)
                    self.dynamic_components.append(table)

        dynamic_page = tgb.Page()
        dynamic_page.add(*self.dynamic_components)
        return dynamic_page


# Example usage
config = DashboardConfig(
    data_points=['total_sales', 'top_products', 'sales_over_time'],
    num_charts=3,
    color_palette=['blue', 'green', 'red', 'purple']
)

sales_pipeline_results = {
    'data_points': {
        'total_sales': 100000,
        'top_products': [{'product_line': 'A', 'total': 50000}, {'product_line': 'B', 'total': 30000}],
        'sales_over_time': [{'date': '2023-01', 'total': 10000}, {'date': '2023-02', 'total': 20000}]
    }
}

data_point_mapping = {
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

# dashboard_generator = DashboardGenerator(config=config,
#                                          sales_pipeline_results=sales_pipeline_results,
#                                          data_point_mapping=DATA_POINT_MAPPING_V2)
# dynamic_page = dashboard_generator.generate_components()
#
# print(dynamic_page)
