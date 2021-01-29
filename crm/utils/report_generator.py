import os
import sys
from dataclasses import dataclass
from typing import List

from crm.base.state import State
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import rx
from rx import scheduler


@dataclass
class ReportOptions:
    title: str = "Simulation Results"
    name: str = "Default report name"

    shutdown_button: bool = True


class ReportGenerator:
    """
    Generate a dash html report
    """

    def __init__(self, options: ReportOptions):
        self.options = options

    def generate_report(self, states: List[State]):
        app = self.create_dash_layout()
        app.run_server()

    def create_dash_layout(self):
        app = dash.Dash(self.options.name)

        app.layout = html.Div(id="container", children=[
            html.H1(children=self.options.title),

            html.Button("Shutdown", id="shutdown-button", n_clicks=0) if self.options.shutdown_button else None,
            html.Div(id="status"),
        ])

        @app.callback(
            dash.dependencies.Output("status", "children"),
            [dash.dependencies.Input('shutdown-button', 'n_clicks')]
        )
        def shutdown_callback(n_clicks):
            if n_clicks > 0:
                rx.timer(1.0).subscribe(lambda _: os._exit(0), scheduler=scheduler.NewThreadScheduler())
                return "Report server has shutdown"

        return app
