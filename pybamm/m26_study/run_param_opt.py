import pybamm
from pybamm.m26_study.pybamm_sim import pybamm_sim
from pybamm.m26_study.m26_params import get_parameter_values

import plotly.express as px
from plotly.express.colors import sample_colorscale
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error as rmse
from scipy.interpolate import interp1d

colors = px.colors.qualitative.Plotly


def optimize_for_rpt(
    rpt_num,
    cycle_df,
    params0,
    params_range,
    step_num_start=6 + 6,
    step_num_end=6 + 6 + 2,
):
    plot_df = cycle_df[(cycle_df["nodeID"] == rpt_num)]

    opt_input_df = plot_df[
        (plot_df["step_number"] >= step_num_start)
        & (plot_df["step_number"] <= step_num_end)
    ].copy()
    opt_input_df["time"] = (
        opt_input_df["total_time_s"] - opt_input_df["total_time_s"].iloc[0]
    )
    opt_input_df["current"] = -np.around(opt_input_df["current"], 2)
    opt_input_df["voltage"] = np.around(opt_input_df["voltage"], 3)

    start_voltage = plot_df.loc[opt_input_df.index[0] - 1, "voltage"]

    t_arr = np.arange(0, opt_input_df.time.max(), 10)
    opt_input_dict = {
        "time": t_arr,
        "current": interp1d(x=opt_input_df.time, y=opt_input_df.current)(t_arr),
        "voltage": interp1d(x=opt_input_df.time, y=opt_input_df.voltage)(t_arr),
    }

    options = {
        "SEI": "reaction limited",
        "SEI porosity change": "true",
        "open-circuit potential": ("current sigmoid", "current sigmoid"),
    }
    sim = pybamm_sim(options=options, parameters=params0, save=False)
    results_df, opt_results = sim.optimize_params(
        params_range=params_range,
        opt_input_dict=opt_input_dict,
        start_voltage=start_voltage,
        method="Powell"
        # method="Nelder-Mead"
    )
    return results_df, opt_results


def optimize_for_cell(cell_id):
    pass
