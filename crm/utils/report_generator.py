import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import numpy as np
from crm.base.solver import SolverMeta
from crm.base.state import State
from crm.base.system_spec import SystemSpec
from crm.utils.pandas import StateDataFrame
from crm.utils.csd_grid import edges_to_center_grid
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

    csd_edges: np.ndarray = np.linspace(0, 300e-6, 100)
    csd_logx: bool = True
    csd_logy: bool = True

    debug: bool = False


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

    def create_temperature_profile(self, sdf: StateDataFrame):
        if not self.options.temperature_profile:
            return None

        fig = self._timeseries_plot(sdf.temperature, plot_fcn=px.line,
                                    ylabel="Temperature (degc)")
        return html.Div(children=[
            html.H3("Temperature Profile"),
            dcc.Graph(figure=fig)
        ])

    def create_concentration_profile(self, sdf: StateDataFrame):
        if not self.options.concentration_profile:
            return None
        df = sdf.concentration
        if self.options.solubility_in_concentration_profile:
            sols = sdf.solubility

            df = pd.concat([df, sols], axis=1)

        fig_conc: go.Figure = self._timeseries_plot(df, plot_fcn=px.line,
                                                    ylabel="Concentration (g/g)")
        children = [
            html.H3("Concentration Profile"),
            dcc.Graph(figure=fig_conc)
        ]
        if self.options.supersaturation_profile:
            df = sdf.supersaturation
            fig_ss = self._timeseries_plot(df, ylabel="Supersaturation", plot_fcn=px.line)
            children.extend([
                html.H3("Supersaturation Profile"),
                dcc.Graph(figure=fig_ss)
            ])
        return html.Div(children=children)

    def create_volume_fraction_profile(self, sdf: StateDataFrame):
        vfs = sdf.volume_fraction
        fig = self._timeseries_plot(vfs * 100, plot_fcn=px.line, ylabel="%")
        return html.Div(children=[
            html.H3("Volume Fraction Profile"),
            dcc.Graph(figure=fig)
        ])

    def create_particle_count_profile(self, sdf: StateDataFrame):
        if not self.options.particle_count_profile:
            return None

        counts = sdf.counts
        fig = self._timeseries_plot(counts, plot_fcn=px.line, ylabel="Count (#/m^3)")
        fig.update_yaxes(type="log")
        return html.Div(children=[
            html.H3("Particle Count Profile"),
            dcc.Graph(figure=fig)
        ])

    def create_particle_size_profile(self, sdf: StateDataFrame):
        if not self.options.particle_size_profile:
            return None

        sizes = sdf.quantiles
        sizes = self._multiindex_to_flatten_index(sizes)

        fig = self._timeseries_plot(sizes, plot_fcn=px.line, ylabel="Size (m)")
        fig.update_yaxes(type="log")
        return html.Div(children=[
            html.H3("Particle Size Profile"),
            dcc.Graph(figure=fig)
        ])

    def create_kinetics_profile(self, sdf: StateDataFrame):
        nucleation = sdf.nucleation_rates
        nucleation = self._multiindex_to_flatten_index(nucleation)

        gds = sdf.gds
        gds = self._multiindex_to_flatten_index(gds)

        fig_nuc = self._timeseries_plot(nucleation, plot_fcn=px.line, ylabel="Count (#/m^3/s)")
        fig_nuc.update_yaxes(type="log")

        fig_gds = self._timeseries_plot(gds, plot_fcn=px.line, ylabel="GD (m/s)")

        return html.Div(children=[
            html.H3("Nucleation Rates"),
            dcc.Graph(figure=fig_nuc),
            html.H3("Growth/Dissolution Rates"),
            dcc.Graph(figure=fig_gds),
        ])

    def create_csd(self, sdf: StateDataFrame, app):
        time = sdf.time
        edges = self.options.csd_edges
        logx = self.options.csd_logx
        logy = self.options.csd_logy
        csds = sdf.get_csd(edges)
        grids = edges_to_center_grid(edges)
        children = html.Div(children=[
            dcc.Slider(id="csd-slider", min=time.min(), max=time.max(), value=time.min(), step=1, tooltip=dict(always_visible=True)),
            dcc.Graph(id="csd"),
        ])

        @app.callback(
            dash.dependencies.Output("csd", "figure"),
            [dash.dependencies.Input("csd-slider", "value")]
        )
        def func(slider_value):
            idx = time.get_loc(slider_value, method="nearest")
            t = time[idx]
            csd = csds.loc[t]
            df = pd.DataFrame(data=csd.tolist(), columns=grids, index=csds.columns).T
            fig = px.line(df)
            fig.update_layout(xaxis_title="Size (m)", yaxis_title="Count (#/m^3)", )

            if logx:
                fig.update_xaxes(type="log")
            if logy:
                fig.update_yaxes(type="log")
            return fig

        return children

    def create_result(self, sdf: StateDataFrame, app):
        return html.Div(id="result-container", children=[
            html.H2("Bulk Properties"),
            self.create_temperature_profile(sdf),
            self.create_concentration_profile(sdf),
            self.create_volume_fraction_profile(sdf),

            html.H2("Particle Properties"),
            self.create_particle_count_profile(sdf),
            self.create_particle_size_profile(sdf),

            html.H2("Kinetics"),
            self.create_kinetics_profile(sdf),

            html.H2("CSD"),
            self.create_csd(sdf, app),
        ])

    def create_dash_layout(self, states: List[State]):
        sdf = StateDataFrame(states)
        app = dash.Dash(self.options.name)

        app.layout = html.Div(id="container", children=[
            html.H1(children=self.options.title),
            self.create_meta(),

            self.create_result(sdf, app),

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

    def _multiindex_to_flatten_index(self, df):
        flat_index = df.columns.to_flat_index()
        flat_index = ["_".join([str(x) for x in f]) for f in flat_index]
        df.columns = flat_index
        return df
