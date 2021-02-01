import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Callable
import numpy as np
from crm.base.solver import SolverMeta
from crm.base.state import State
from crm.base.system_spec import SystemSpec
from crm.utils.pandas import state_list_to_dataframes
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import rx
from rx import scheduler
import dash_table
import pandas as pd


@dataclass
class ReportOptions:
    title: str = "Simulation Results"
    name: str = "Default report name"

    shutdown_button: bool = True

    meta: bool = True

    temperature_profile: bool = True

    concentration_profile: bool = True
    solubility_in_concentration_profile: bool = True
    compute_solubility: bool = False
    supersaturation_profile: bool = True

    particle_count_profile: bool = True

    particle_size_profile: bool = True

    debug: bool = False


DictDfType = Dict[str, pd.DataFrame]


class ReportGenerator:
    """
    Generate a dash html report
    """

    def __init__(self, options: ReportOptions, meta: SolverMeta = None, system_spec: SystemSpec = None):
        self.system_spec = system_spec
        self.meta = meta
        self.options = options

    def generate_report(self, states: List[State]):
        """
        Public interface to generate report.
        :param states:
        :return:
        """
        app = self.create_dash_layout(states)
        app.run_server(debug=self.options.debug)

    def create_meta(self):
        """
        Create meta panel
        :return: dash html element
        """
        if self.meta is None or not self.options.meta:
            return None

        d = self.meta.__dict__
        df = pd.DataFrame.from_dict(d, orient="index", columns=["Value"]).rename_axis("Key").reset_index()
        return html.Div(id="meta", children=[
            html.H2("Solver Metadata"),
            dash_table.DataTable(
                id="meta-table",
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
            )
        ])

    def create_temperature_profile(self, dict_df: DictDfType):
        if not self.options.temperature_profile:
            return None

        fig = self._timeseries_plot(dict_df["ct"]["temperature"], plot_fcn=px.line,
                                    ylabel="Temperature (degc)")
        return html.Div(children=[
            html.H3("Temperature Profile"),
            dcc.Graph(figure=fig)
        ])

    def create_concentration_profile(self, dict_df: DictDfType):
        if not self.options.concentration_profile:
            return None

        df = dict_df["ct"]["concentration"]

        if self.options.solubility_in_concentration_profile:
            if not "sol" in dict_df:
                if self.options.compute_solubility:
                    t = dict_df["ct"]["temperature"]
                    sols = [f.solubility(t) for f in self.system_spec.forms]
                else:
                    sols = None
            else:
                sols = dict_df["sol"].loc[:, "sol"]

            df = pd.concat([df, sols], axis=1)

        fig_conc: go.Figure = self._timeseries_plot(df, plot_fcn=px.line,
                                                    ylabel="Concentration (g/g)")
        children = [
            html.H3("Concentration Profile"),
            dcc.Graph(figure=fig_conc)
        ]
        if self.options.supersaturation_profile:
            df = dict_df["sol"].loc[:, "ss"]
            fig_ss = self._timeseries_plot(df, ylabel="Supersaturation", plot_fcn=px.line)
            children.extend([
                html.H3("Supersaturation Profile"),
                dcc.Graph(figure=fig_ss)
            ])
        return html.Div(children=children)

    def create_particle_count_profile(self, dict_df):
        if not self.options.particle_count_profile:
            return None

        n = dict_df["n"][["n"]]

        def compute_count(x: pd.Series):
            return x.apply(lambda xx: xx[:, -1].sum())

        counts = n.apply(compute_count)
        counts = counts.droplevel(0, axis=1)
        fig = self._timeseries_plot(counts, plot_fcn=px.line, ylabel="Count (#/m^3)")
        fig.update_yaxes(type="log")
        return html.Div(children=[
            html.H3("Particle Count Profile"),
            dcc.Graph(figure=fig)
        ])

    def create_particle_size_profile(self, dict_df):
        if not self.options.particle_size_profile:
            return None

        n = dict_df["n"][["n"]]
        n = n.droplevel(0, axis=1)
        dfs = []

        def compute_quantile(x: np.ndarray, names):
            if x.shape[0] == 0:
                q = [None, None, None]
            else:
                q = np.quantile(x[:, 0], [0.1, 0.5, 0.9])
            return pd.Series(q, index=names)

        for i, form_n in n.iteritems():
            names = ["D10", "D50", "D90"]
            names = [f"{i}_{n}" for n in names]
            # ND is not supported yet.
            df = form_n.apply(lambda x: compute_quantile(x, names))
            dfs.append(df)
        sizes = pd.concat(dfs, axis=1)
        sizes.index = n.index

        fig = self._timeseries_plot(sizes, plot_fcn=px.line, ylabel="Size (m)")
        fig.update_yaxes(type="log")
        return html.Div(children=[
            html.H3("Particle Size Profile"),
            dcc.Graph(figure=fig)
        ])

    def create_result(self, dict_df: DictDfType):
        return html.Div(id="result-container", children=[
            html.H2("Bulk Properties"),
            self.create_temperature_profile(dict_df),
            self.create_concentration_profile(dict_df),

            html.H2("Particle Properties"),
            self.create_particle_count_profile(dict_df),
            self.create_particle_size_profile(dict_df),
        ])

    def create_dash_layout(self, states: List[State]):
        dict_df = state_list_to_dataframes(states, self.system_spec)
        app = dash.Dash(self.options.name)

        app.layout = html.Div(id="container", children=[
            html.H1(children=self.options.title),
            self.create_meta(),

            self.create_result(dict_df),

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

    def _timeseries_plot(self, df: pd.DataFrame, plot_fcn: Callable = px.scatter, xlabel: str = "Time (s)",
                         ylabel: str = None):
        fig = plot_fcn(df)
        if xlabel is not None:
            fig.update_layout(xaxis_title=xlabel)
        if ylabel is not None:
            fig.update_layout(yaxis_title=ylabel)
        fig.update_layout(legend_title_text=None)
        return fig
