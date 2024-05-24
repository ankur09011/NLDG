from pydantic import BaseModel, validator, Field
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from taipy.gui import builder as tgb
import numpy as np
import plotly.graph_objects as go


class BaseDataModel(BaseModel):
    result_data: Dict[str, Any] = Field(default_factory=dict)

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

    def create_components(self, config: Dict[str, Any]):
        components = []
        data_points = self.result_data.get('data_points', {})

        for item in config['items']:
            print(item)
            data_point = data_points.get(item['data_point'])
            print(data_point)
            if not data_point:
                continue

            if item['type'] == 'heading':
                text_component = tgb.text(f"## {item['label']}", mode="md")
                components.append(text_component)

            elif item['type'] == 'bar_chart':
                print(item)
                bar_chart_df = pd.DataFrame(data_point)
                print(bar_chart_df)
                bar_chart = tgb.chart("{bar_chart_df}", type="bar",
                                      x=item['x'],
                                      y=item['y'])

                # fig = go.Figure(data=go.Bar(x=df[item['x']], y=df[item['y']]))
                # fig.update_layout(title=item['title'])
                # bar_chart = tgb.chart(figure="{fig}")
                components.append(bar_chart)

            elif item['type'] == 'line_chart':
                df = pd.DataFrame(data_point)
                fig = go.Figure(data=[go.Scatter(x=df[item['x']],
                                                 y=df[item['y']],
                                                 mode='lines+markers')])
                fig.update_layout(title=item['title'])
                line_chart = tgb.chart(figure="{fig}")
                components.append(line_chart)

            elif item['type'] == 'table':
                df = pd.DataFrame(data_point)
                table = tgb.table("{df}", page_size=50)
                components.append(table)

        return components


class BaseDocument(BaseModel):
    text: str
    metadata: Dict[str, Any]
    smart_connects: List[str]  # References to SmartConnect objects by their IDs


