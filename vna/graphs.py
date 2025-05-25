import os
import random
from datetime import datetime
from itertools import product

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from skrf.io import touchstone
from skrf import plotting, Network, figure

from vna.VNA_defaults import DEFAULT_FIGURE_SIZE, DEFAULT_FILE_TYPE, DEFAULT_COLOUR_MAP

matplotlib.use("TkAgg")
sns.set_theme(style="whitegrid", font_scale=2)

from vna.VNA_utils import (
    convert_magnitude_rows_to_db,
    ghz_to_hz,
    get_frequency_column_headings_list,
    hz_to_ghz,
    convert_magnitude_to_db,
    hz_to_mhz,
    open_pickled_object,
    get_graph_path,
)
from vna.VNA_enums import (
    ConfusionMatrixKey,
    DataFrameCols,
    MeasurementKey,
    MagnitudeOrPhase,
    SParam,
)


def svm_vs_dt_strip_plot(results_df: pd.DataFrame):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]

    melted = pd.melt(
        accuracy_df, id_vars=["label", "classifier"], value_vars=["precision"]
    )

    fig, ax = plt.subplots()
    sns.despine(bottom=True, left=True)
    sns.stripplot(
        data=melted,
        x="label",
        y="value",
        hue="classifier",
        dodge=True,
        alpha=0.25,
        zorder=1,
        legend=False,
    )

    sns.pointplot(
        data=melted,
        x="label",
        y="value",
        hue="classifier",
        dodge=0.8 - 0.8 / 2,
        palette="dark",
        errorbar=None,
        markers="d",
        markersize=6,
        linestyle="none",
    )
    sns.move_legend(
        ax,
        loc="best",
        ncol=3,
        frameon=True,
        columnspacing=1,
        handletextpad=0,
        title="Classifier",
        labels=["SVM", "Decision Tree"],
    )
    ax.set(
        xlabel="Experiment",
        ylabel="Classifier Accuracy",
        title="SVM vs Decision Tree Classification Accuracy \n For Each Experiment",
    )

    return


def svm_vs_dtree_violin_plot(results_df: pd.DataFrame):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    melted = pd.melt(
        accuracy_df, id_vars=["label", "classifier"], value_vars=["precision"]
    )

    fig, ax = plt.subplots()
    sns.boxplot(data=melted, x="value", y="label", hue="classifier")
    # sns.move_legend(
    #     ax, loc="lower right", ncol=2, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
    #     labels=['SVM', 'D Tree']
    # )
    ax.set(
        ylabel="Experiment",
        xlabel="Classifier Accuracy",
        title="SVM vs Decision Tree Classification Accuracy \n For Each Experiment",
    )
    major_ticks = np.arange(0, 1.1, 0.1)  # Set major ticks every 2 units
    plt.xlim(0, 1)
    plt.xticks(major_ticks)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", bottom=True)
    legend = plt.legend()

    # Access the legend object
    legend.set_title("Classifier")  # Set legend title

    # Set labels
    for text, label in zip(legend.get_texts(), ["SVM", "D Tree"]):
        text.set_text(label)
    return


def full_vs_filtered_features_plot(results_df: pd.DataFrame):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    melted = pd.melt(accuracy_df, id_vars=["full or filtered"], value_vars=["f1-score"])
    fig, ax = plt.subplots()
    sns.boxplot(
        data=melted,
        y="full or filtered",
        x="value",
        hue="full or filtered",
        legend=False,
    )
    ax.set(
        xlabel="Classifier Accuracy",
        ylabel="Feature Set",
        title="Full vs Filtered Features Classification Accuracy",
    )
    major_ticks = np.arange(0, 1.1, 0.1)  # Set major ticks every 2 units
    plt.xlim(0, 1)
    plt.xticks(major_ticks)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", bottom=True)


def freq_band_line_plot(results_df: pd.DataFrame):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    melted = pd.melt(
        accuracy_df,
        id_vars=["label", "low_frequency", "high_frequency"],
        value_vars=["precision"],
    )
    melted["mid_freq"] = round((melted["high_frequency"].astype(float) - 0.05), 2)
    fig, ax = plt.subplots()

    g = sns.lineplot(
        data=melted,
        x="mid_freq",
        y="value",
        hue="label",
        style="label",
        markers=True,
        dashes=False,
        errorbar=None,
    )
    ax.set(
        ylabel="Classifier Accuracy",
        xlabel="Frequency Bands (GHz)",
        title="Mean Classifier Accuracy For Each Tested Frequency Band",
    )
    # ax.xaxis.set_major_locator(mticker.AutoMajorLocator(13))
    major_ticks = np.arange(0, 4, 0.5)  # Set major ticks every 2 units
    plt.xlim(0, 4)
    plt.xticks(major_ticks)
    plt.legend(title="", fontsize=16)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", bottom=True)
    # plt.gca().xaxis.set_minor_locator(mticker.AutoMinorLocator())
    # Set plot title


def select_top_value(group):
    return group.nlargest(1, "value")


def generate_label_from_row(row):
    return f"{row['classifier'].upper()} {row['full or filtered'].split(' ')[0]} {row['type'].title()} {(' ').join(row['s_param'].split('_')).title()} "


def top_classifier_for_each_band(results_df: pd.DataFrame, include_ALL_sparams=False):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    melted = pd.melt(
        accuracy_df,
        id_vars=[
            "label",
            "classifier",
            "full or filtered",
            "type",
            "s_param",
            "low_frequency",
            "high_frequency",
        ],
        value_vars=["precision"],
    )
    melted["mid_freq"] = round((melted["high_frequency"].astype(float) - 0.05), 2)
    title = "The Top Performing Classifier Accuracy For Each Frequency Band \n With All S Parameter Measurements Included"
    if not include_ALL_sparams:
        melted = melted[~melted["s_param"].isin(["all_Sparams"])]
        title = "The Top Performing Classifier Accuracy For Each Frequency Band \n With All S Parameter Measurements Removed"
    grouped = melted.groupby(["label", "mid_freq"], as_index=False)
    top_values = grouped.apply(select_top_value)
    top_values.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots()

    sns.lineplot(
        data=top_values,
        x="mid_freq",
        y="value",
        hue="label",
        style="label",
        markers=True,
        dashes=False,
    )
    ax.set(ylabel="Classifier Accuracy", xlabel="Frequency Bands (GHz)", title=title)
    # ax.xaxis.set_major_locator(mticker.AutoMajorLocator(13))
    top_top_values = top_values.nlargest(10, "value")
    plt.legend(title="", fontsize=16)
    n = 1
    for index, row in top_top_values.iterrows():
        label = generate_label_from_row(row)
        print(f"{n} & {label.title()} & {row['value']:.2f} \\\\ \\hline")
        mid_freq = row["mid_freq"]
        precision = row["value"]
        plt.text(
            mid_freq,
            precision,
            f"{n}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )
        n += 1

    major_ticks = np.arange(0, 4, 0.5)  # Set major ticks every 2 units
    plt.xlim(0, 4)
    plt.xticks(major_ticks)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", bottom=True)


def fix_uscore_title_case(value):
    return (" ").join(value.split("_")).title()


def max_accuracy_for_mag_sparam_categories(
    results_df: pd.DataFrame, n_to_plot=60, include_all=False
):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    melted = melt_and_filter_mag_sparam(accuracy_df, include_all)

    # top_n_groups = melted.groupby("type")["value"].max().nlargest(n_to_plot).index

    melted_df = melted.sort_values(["value"], ascending=False)[:n_to_plot]
    melted_df["type"] = melted_df["type"].map(lambda x: x.replace("Magnitude", "Mag"))
    fig, ax = plt.subplots()
    sns.despine(bottom=True, left=True)
    pt = sns.stripplot(
        data=melted_df,
        x="type",
        y="value",
        hue="label",
        jitter=True,
        legend=True,
    )
    # sns.move_legend(
    #     ax, loc="lower right", ncol=2, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
    #     labels=['SVM', 'D Tree']
    # )
    ax.set(
        xlabel="Measurement Combination",
        ylabel="Classifier Accuracy",
        title=f"SVM vs Decision Tree Classification Accuracy \n Showing The Top {n_to_plot} S Parameter Combinations By Max Accuracy",
    )
    plt.legend(title="", fontsize=16)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", bottom=True)
    pt.tick_params(labelsize=12)


def melt_and_filter_mag_sparam(accuracy_df, include_all=False):
    # combine mag or phase and sparam
    accuracy_df.loc[:, "type"] = accuracy_df["type"] + "_" + accuracy_df["s_param"]
    melted = pd.melt(accuracy_df, id_vars=["label", "type"], value_vars=["precision"])
    # melted[["type", "s_param"]].apply(tuple, axis=1)
    if not include_all:
        # inverting returned boolean df to remove these (~ is NOT)
        # removes the "all" category
        melted = melted[
            ~melted["type"].isin(["magnitude_all_Sparams", "phase_all_Sparams"])
        ]
    # fix the titles of the graph
    melted.loc[:, "type"] = melted["type"].apply(fix_uscore_title_case)
    return melted


def get_mean_frequency_band_from_results(results_df):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    return hz_to_mhz(
        ghz_to_hz(
            (
                accuracy_df["high_frequency"].astype(float)
                - accuracy_df["low_frequency"].astype(float)
            ).mean()
        )
    )


def bar_graph_accuracy_comparison(results_df, n_to_plot=6, include_all=False):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    band = round(get_mean_frequency_band_from_results(accuracy_df) / 5) * 5

    melted = melt_and_filter_mag_sparam(accuracy_df, include_all)
    melted["type"] = melted["type"].str.replace("Magnitude ", "|") + "|"
    top_n_groups = melted.groupby("type")["value"].mean().nlargest(n_to_plot).index
    melted_df = melted[melted["type"].isin(top_n_groups)]
    fig, ax = plt.subplots()
    sns.boxplot(data=melted_df, x="value", y="type", hue="label")
    # sns.move_legend(
    #     ax, loc="lower right", ncol=2, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
    #     labels=['SVM', 'D Tree']
    # )
    ax.set(
        xlabel="Classifier Accuracy",
        ylabel="",
        title=f"Comparison of Dipole vs Glove Antenna Far Field \n Wireless Readout Using {melted['type'][0]} and a {band}MHz Channel",
    )

    legend = plt.legend(loc="best")
    plt.show()


def plot_s_param_mag_phase_from_touchstone(touchstone_path, name):
    network = touchstone.hfss_touchstone_2_network(touchstone_path)
    network.name = name
    # network.plot_it_all()
    plt.subplots(1, 2)
    plt.title(name)
    ax = plt.subplot(1, 2, 1)
    plt.title(f"LogMag")
    network.plot_s_db()
    ax.get_legend().remove()

    ax = plt.subplot(1, 2, 2)
    plt.title("Phase")
    network.plot_s_deg()
    ax.legend(loc=(1.04, 0))


def plot_magnitude_from_touchstone(touchstone_path, name):
    plt.figure()
    network: Network = touchstone.hfss_touchstone_2_network(touchstone_path)

    plt.title(name)
    plt.legend([])
    network.plot_s_db()


def subplot_params(network: Network):
    plotting.subplot_params(network)


def plot_sampling_freq(sampling_freq_results):

    fig, ax = plt.subplots()

    ax.set(
        yscale="log",
        xscale="log",
        title="The Calculated Sweep Period For Each Of The Possible \n Bandwidth and Number Of Points Settings On The Pico VNA 6",
    )
    plot = sns.lineplot(
        data=sampling_freq_results,
        x="Number of Points",
        y="Calculated Sampling Frequency (Hz)",
        hue="Bandwidth (Hz)",
        style="Bandwidth (Hz)",
        markers=True,
        dashes=False,
        palette="tab10",
    )
    plt.setp(plot.get_legend().get_texts(), fontsize="12")
    plt.setp(plot.get_legend().get_title(), fontsize="12")


def calulate_sweep_time(
    bandwidth,
    n_points,
    time_per_point=167e-6,
    bandwidth_settle_factor=1.91,
    rearm_time=6.5e-3,
):
    return (
        n_points * (time_per_point + bandwidth_settle_factor / bandwidth) + rearm_time
    )


def gen_sweep_time_df(
    n_points=None,
    bandwidths: [int] = None,
    time_per_point=167e-6,
    bandwidth_settle_factor=1.91,
    rearm_time=6.5e-3,
):
    if n_points is None:
        n_points = [101, 201, 301, 501, 1001, 2001]
    if bandwidths is None:
        bandwidths = [10, 100, 1000, 10_000, 75_000, 140_000]
    combinations = product(n_points, bandwidths)
    output_dict = {}
    output_dict["Number of Points"] = []
    output_dict["Bandwidth (Hz)"] = []
    output_dict["Calculated Sweep Time (s)"] = []
    output_dict["Calculated Sampling Frequency (Hz)"] = []
    for n_point, bandwidth in combinations:
        output_dict["Number of Points"].append(n_point)
        output_dict["Bandwidth (Hz)"].append(bandwidth)
        sweep_time = calulate_sweep_time(
            bandwidth, n_point, time_per_point, bandwidth_settle_factor, rearm_time
        )
        output_dict["Calculated Sweep Time (s)"].append(sweep_time)
        output_dict["Calculated Sampling Frequency (Hz)"].append(1 / sweep_time)

    return pd.DataFrame.from_dict(output_dict)


def make_confusion_matrix_dict_string_from_series(series: pd.Series) -> str:
    return ("_").join(
        [
            series["s_param"],
            series["type"],
            series["low_frequency"],
            series["high_frequency"],
        ]
    )


def confusion_matrix_from_single_result(
    single_result_series: pd.Series,
    labels,
    confusion_matrix_dict,
    confusion_matrix_option: ConfusionMatrixKey,
    convert_to_percent_of_true_labels=False,
) -> None:

    confusion_matrix_key = make_confusion_matrix_dict_string_from_series(
        single_result_series
    )
    confusion_matrix = confusion_matrix_dict[confusion_matrix_key][
        confusion_matrix_option.value
    ]
    if convert_to_percent_of_true_labels:
        confusion_matrix = np.round(
            confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) * 100, 0
        ).astype(int)

    ConfusionMatrixDisplay(confusion_matrix, display_labels=labels).plot()
    plt.title(
        f'Confusion Matrix Using {single_result_series["classifier"].title()} classifier \n'
        f'Between {single_result_series["low_frequency"]} and {single_result_series["high_frequency"]} GHz'
    )
    plt.show()
    return


def display_confusion_matrix_for_top_n_values(
    full_df,
    results_df,
    confusion_matrix_dict,
    confusion_matrix_option=ConfusionMatrixKey.FILTERED_SVM,
    n=1,
    convert_to_percent_of_true_labels=False,
):
    # full_df.columns = [
    #     pd.to_numeric(col, errors="coerce") if col.isnumeric() else col
    #     for col in full_df.columns
    # ]

    accuracy_df = results_df[(results_df["gesture"] == "accuracy")]
    accuracy_df = accuracy_df.sort_values(by="f1-score", ascending=False)
    mag_df = accuracy_df[accuracy_df["type"] == "magnitude"]
    labels = sorted(list(set([label[-1] for label in full_df["label"].unique()])))

    for i in range(n):
        top_magnitude = mag_df.iloc[i]
        confusion_matrix_from_single_result(
            top_magnitude,
            labels,
            confusion_matrix_dict,
            confusion_matrix_option,
            convert_to_percent_of_true_labels,
        )


def plot_fq_time_series(
    single_gesture_data_frame: pd.DataFrame,
    *,
    s_parameter: SParam = None,
    mag_or_phase: MagnitudeOrPhase = None,
    label=None,
    n_random_ids=1,
    target_frequency=None,
):
    """
    Takes in a data frame which contains the results for a single gesture from a single experiment,
    provide the parameters you want to plot and a label for the plot.
    Args:
        single_gesture_data_frame:
        s_parameter:
        mag_or_phase:
        label:
        n_random_ids:
        target_frequency:

    Returns:

    """
    if target_frequency is None:
        raise AttributeError("No target frequency")

    if s_parameter is None or mag_or_phase is None or label is None:
        raise AttributeError(
            f"Must include all params s_param={s_parameter}, mag_or_phase={mag_or_phase}, label={label}"
        )

    filtered_df = single_gesture_data_frame.query(
        f's_parameter == "{s_parameter.value}" and mag_or_phase == "{mag_or_phase.value}" and label == "{label}"'
    )
    grouped_by_id = filtered_df.groupby("id")
    filtered_dfs = []
    random_ids = random.sample(list(grouped_by_id.groups.keys()), n_random_ids)
    random_experiment_list = [filtered_df.query(f'id == "{id}"') for id in random_ids]
    for random_experiment_df in random_experiment_list:
        filtered_dfs.append(filter_fq_cols(random_experiment_df, target_frequency))

    fig, ax = plt.subplots()

    for filtered_df in filtered_dfs:
        closest_fq = get_closest_freq_column(filtered_df, target_frequency)
        ax.plot(
            filtered_df[DataFrameCols.TIME.value],
            filtered_df[closest_fq].apply(convert_magnitude_to_db),
        )
    ax.set_ylabel(f"|{s_parameter}|")
    ax.set_xlabel("Time (s)")
    plt.title(f"|{s_parameter}| Over Time at {hz_to_ghz(closest_fq)} GHz")
    plt.show()


def plot_fq_time_series_as_subplot(
    ax: Axes,
    single_gesture_data_frame: pd.DataFrame,
    *,
    s_parameter: SParam = None,
    mag_or_phase: MagnitudeOrPhase = None,
    label=None,
    n_random_ids=1,
    target_frequency=None,
    color="blue",
    title=False,
):
    """
    Takes in a data frame which contains the results for a single gesture from a single experiment,
    provide the parameters you want to plot and a label for the plot.
    Args:
        single_gesture_data_frame:
        s_parameter:
        mag_or_phase:
        label:
        n_random_ids:
        target_frequency:

    Returns:

    """
    if target_frequency is None:
        raise AttributeError("No target frequency")

    if s_parameter is None or mag_or_phase is None or label is None:
        raise AttributeError(
            f"Must include all params s_param={s_parameter}, mag_or_phase={mag_or_phase}, label={label}"
        )

    filtered_df = single_gesture_data_frame.query(
        f's_parameter == "{s_parameter.value}" and mag_or_phase == "{mag_or_phase.value}" and label == "{label}"'
    )
    grouped_by_id = filtered_df.groupby("id")
    gesture = label.split("_")[-1]
    filtered_dfs = []
    random_ids = random.sample(list(grouped_by_id.groups.keys()), n_random_ids)
    random_experiment_list = [filtered_df.query(f'id == "{id}"') for id in random_ids]

    for random_experiment_df in random_experiment_list:
        filtered_dfs.append(filter_fq_cols(random_experiment_df, target_frequency))

    for filtered_df in filtered_dfs:
        closest_fq = get_closest_freq_column(filtered_df, target_frequency)
        if mag_or_phase == MagnitudeOrPhase.Magnitude:
            time_series = filtered_df[closest_fq].apply(convert_magnitude_to_db)
        else:
            time_series = filtered_df[closest_fq]
        ax.plot(
            filtered_df[DataFrameCols.TIME.value],
            time_series,
            label=gesture,
            color=color,
        )
    if title:
        ax.set_title(f"{gesture}")
    return ax


#
# def add_fig_legend(fig, group_labels, loc="lower center", ncol=4):
#     # Get colours for current style
#     colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     # Set up handles (the bits that are drawn in the legend)
#     handles = []
#     for group_idx in range(len(group_labels)):
#         # Create a simple patch that is the correct colour
#         colour = colours[group_idx]
#         handles.append(Patch(edgecolor=colour, facecolor=colour, fill=True))
#     # Acutally create our figure legend, using the handles and labels
#     fig.legend(handles=handles, labels=group_labels, loc=loc, ncol=ncol)


def plot_multiple_gestures_on_time_series(
    *,
    data_frame: pd.DataFrame,
    experiment_label,
    gestures,
    target_s_param: SParam,
    mag_or_phase: MagnitudeOrPhase = MagnitudeOrPhase.Magnitude,
    target_frequency,
    n_random_ids=1,
    c_map_label=DEFAULT_COLOUR_MAP,
    filetype=DEFAULT_FILE_TYPE,
    fname=None,
    file_output_path=None,
    save_to_file=False,
):
    c_map = get_cmap(c_map_label)
    plot_labels = gestures
    fig, axes = plt.subplots(nrows=len(plot_labels), ncols=1, sharex=True, sharey=True)
    fig.suptitle(
        f"|{target_s_param.value}| Over Time at {hz_to_ghz(target_frequency)} GHz"
    )
    if len(plot_labels) > 1:
        axes = axes.flat
    plotted_axes = []
    for i, (plot_label, ax) in enumerate(zip(plot_labels, axes)):
        indv_figure, indv_axis = plt.subplots(1, 1)
        color = c_map(i / len(plot_labels))
        plotted_axes.append(
            plot_fq_time_series_as_subplot(
                ax,
                data_frame,
                s_parameter=target_s_param,
                mag_or_phase=mag_or_phase,
                label=plot_label,
                n_random_ids=n_random_ids,
                target_frequency=target_frequency,
                color=color,
            )
        )
        plot_fq_time_series_as_subplot(
            indv_axis,
            data_frame,
            s_parameter=target_s_param,
            mag_or_phase=mag_or_phase,
            label=plot_label,
            n_random_ids=n_random_ids,
            target_frequency=target_frequency,
            color=color,
        )
        indv_figure.show()

    collected_handles, collected_labels = [], []
    for ax in plotted_axes:
        handles, labels = ax.get_legend_handles_labels()
        collected_handles.extend(handles)
        collected_labels.extend(labels)

    fig.supxlabel("Time (s)")
    fig.supylabel(f"|{target_s_param.value}|")
    fig.legend(handles=collected_handles, labels=collected_labels)
    # plt.tight_layout()
    if save_to_file:
        if fname is None:
            fname = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}-{target_s_param.value}-{mag_or_phase.value}-{c_map_label}.{filetype}"
        if file_output_path is None:
            file_output_path = get_graph_path()
        if experiment_label:
            file_output_path = os.path.join(file_output_path, experiment_label)
            os.makedirs(file_output_path, exist_ok=True)
        file_output_path = os.path.join(
            file_output_path, "Time Series", mag_or_phase.value, target_s_param.value
        )
        os.makedirs(file_output_path, exist_ok=True)
        plt.savefig(os.path.join(file_output_path, fname), format=filetype)

    return fig


def filter_fq_cols(df, target_frequency):
    closest_fq_col = get_closest_freq_column(df, target_frequency)
    # this filters all teh columns
    # fq_cols = [int(col) for col in [col for col in test.columns.astype(str) if re.search(r'\d', col)]]
    str_cols = [col for col in df.columns if type(col) is str]
    str_cols.append(closest_fq_col)
    return df[str_cols]


def get_closest_freq_column(data_frame, target_frequency):
    fq_series = get_frequency_column_headings_list(data_frame)
    closest_fq_col = fq_series[
        (np.abs(np.asarray(fq_series) - target_frequency)).argmin()
    ]
    return closest_fq_col


def plot_comparison_table(
    full_df, *, s_parameter=None, mag_or_phase=None, target_frequency=None
):
    if target_frequency is None:
        raise AttributeError("No target frequency")

    if s_parameter is None or mag_or_phase is None:
        raise AttributeError(
            f"Must include all params s_param={s_parameter}, mag_or_phase={mag_or_phase}"
        )

    filtered_df = full_df.query(
        f's_parameter == "{s_parameter}" and mag_or_phase == "{mag_or_phase}"'
    )

    grouped_by_label = filtered_df.groupby("id")
    dfs_to_plot = []
    for label_group in grouped_by_label:

        grouped_by_ids = label_group.groupby("id")
        random_test = random.choice(list(grouped_by_ids.groups.keys()))
        dfs_to_plot.append(
            filter_fq_cols(
                filtered_df.query(f'id == "{random_test}"'), target_frequency
            )
        )

    fig, axs = plt.subplots(nrows=len(dfs_to_plot), ncols=1, sharex=True)

    for ax, df in zip(axs, dfs_to_plot):
        closest_fq = get_closest_freq_column(df, target_frequency)
        ax.plot(df[DataFrameCols.TIME.value], df[closest_fq])
        ax.set_ylabel(f"|{s_parameter}|")
        ax.set_xlabel("Time (s)")
        plt.title(f"|{s_parameter}| Over Time at {hz_to_ghz(closest_fq)} GHz")
        plt.show()


def display_confusion_matrix_for_given_accuracy_value(
    full_df,
    results_df,
    confusion_matrix_dict,
    accuracy_value,
    measurement: MeasurementKey,
    confusion_matrix_option: ConfusionMatrixKey,
) -> None:
    # allows you to index the lowest values by using -ve numbers
    if accuracy_value > 0:
        accuracy_index = accuracy_value - 1
    else:
        accuracy_index = accuracy_value

    full_df.columns = [
        pd.to_numeric(col, errors="coerce") if col.isnumeric() else col
        for col in full_df.columns
    ]

    accuracy_df = results_df[(results_df["gesture"] == "accuracy")]
    accuracy_df = accuracy_df.sort_values(by="f1-score", ascending=False)
    measurement_df = accuracy_df[accuracy_df["type"] == measurement.value]
    selected_result = measurement_df.iloc[accuracy_index]

    labels = [label[-1] for label in full_df["label"].unique()]
    confusion_matrix_from_single_result(
        selected_result, labels, confusion_matrix_dict, confusion_matrix_option
    )


def show_top_five_and_bottom_five_confusion_matricies(
    full_df,
    results_df,
    confusion_matrix_dict,
    measurement: MeasurementKey,
    confusion_matrix_option: ConfusionMatrixKey,
) -> None:
    for i in range(5):
        display_confusion_matrix_for_given_accuracy_value(
            full_df,
            results_df,
            confusion_matrix_dict,
            i,
            measurement,
            confusion_matrix_option,
        )
    for i in range(-5, -1):
        display_confusion_matrix_for_given_accuracy_value(
            full_df,
            results_df,
            confusion_matrix_dict,
            i,
            measurement,
            confusion_matrix_option,
        )


def plot_s_param_channels(
    full_df, target_freq_hz, measurement: MeasurementKey, s_params_to_plot=None
):

    full_df = full_df[full_df["mag_or_phase"] == measurement.value]

    # find closest fq to target
    closest_fq = find_nearest_frequency(full_df, target_freq_hz)
    filter_id = full_df["id"].unique()[0]
    filtered_df = full_df[full_df["id"] == filter_id]
    if s_params_to_plot is None:
        s_params_to_plot = filtered_df["s_parameter"].unique()
    fig, ax = plt.subplots(len(s_params_to_plot), 1, sharex=True)
    ax = ax.flatten()
    for axis, s_param in zip(ax, s_params_to_plot):
        s_param_df = filtered_df[filtered_df["s_parameter"] == s_param]
        sns.lineplot(data=s_param_df, x="time", y=closest_fq, ax=axis)
        axis.set_title(f"{s_param}")
    plt.show()


def find_nearest_frequency(full_df, target_frequency_hz):
    array = np.asarray(list(full_df.columns)[5:])
    idx = (np.abs(array - target_frequency_hz)).argmin()
    return array[idx]


def create_data_normaliser(data_to_normalise):
    """
    Takes in a data series to plot and returns an object which will normalise any data to
    that range
    Args:
        data_to_normalise:
        Data series or array which contains the data to be normalised
    Returns:

    """
    max_value = data_to_normalise.max().max()
    min_value = data_to_normalise.min().min()
    return matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)


def plot_3d_time_series(
    data_frame_to_plot: pd.DataFrame,
    c_map_name=DEFAULT_COLOUR_MAP,
    fig_size=DEFAULT_FIGURE_SIZE,
    file_output_path=None,
    file_output_flag=False,
    fname=None,
    experiment_label=None,
    filetype=DEFAULT_FILE_TYPE,
    font_size=14,
):
    """
    Plots a 3D time series from a results dataframe.

    Args:
        data_frame_to_plot: DataFrame with the data to plot.
        seaborn_style: Optional seaborn style to use for the plot.
        fig_size: Size of the figure.

    Returns:
        fig: The generated Matplotlib figure object.
    """

    # Set global font size for all text elements
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )

    phase_or_mag = data_frame_to_plot["mag_or_phase"].iloc[0]

    if phase_or_mag == MagnitudeOrPhase.Phase.value:
        data_frame_to_plot.iloc[:, 5:] = data_frame_to_plot.iloc[:, 5:] * 180 / np.pi

    cmap = matplotlib.colormaps[c_map_name]

    s_parameter = data_frame_to_plot["s_parameter"].iloc[0]
    gesture = data_frame_to_plot["label"].iloc[0].split("_")[-1]

    group = data_frame_to_plot.iloc[:, 4:].groupby("time")
    frequency_cols = list(
        map(int, map(hz_to_mhz, list(data_frame_to_plot.columns[5:])))
    )

    plotting_indexes = np.linspace(0, data_frame_to_plot["time"].max(), len(group))
    time = data_frame_to_plot["time"]
    times_normalized = time - time.min()

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(projection="3d")

    data_to_plot = data_frame_to_plot.iloc[:, 5:]
    data_normaliser = create_data_normaliser(data_to_plot)

    # For collecting axis limits
    all_frequency, all_magnitude, all_times = [], [], []

    for plotting_index, (group_name, df) in reversed(
        list(zip(plotting_indexes, group))
    ):
        magnitude_value = df.iloc[:, 1:].values[0]
        magnitude_array = np.asarray(magnitude_value)
        frequency_array = np.asarray(frequency_cols)

        # Create consecutive segments
        points = np.array([frequency_array, magnitude_array]).T.reshape(-1, 1, 2)
        segments_2d = np.concatenate([points[:-1], points[1:]], axis=1)

        # Convert to 3D segments at fixed z
        z_value = plotting_index
        segments_3d = []
        for seg in segments_2d:
            seg3d = np.column_stack(
                (
                    np.full(seg.shape[0], z_value),  # X = Time
                    seg[:, 0],  # Y = Frequency
                    seg[:, 1],  # Z = Magnitude/Phase
                )
            )
            segments_3d.append(seg3d)
        segments_3d = np.array(segments_3d)

        # Color per segment based on midpoint of y-values
        magnitude_segment_mids = 0.5 * (magnitude_array[:-1] + magnitude_array[1:])
        color_values = cmap(data_normaliser(magnitude_segment_mids))

        # Create and add collection
        line_collection = Line3DCollection(
            segments_3d, colors=color_values, linewidths=2
        )
        ax.add_collection3d(line_collection)

        # Collect points for setting limits
        all_frequency.extend(frequency_array)
        all_magnitude.extend(magnitude_array)
        all_times.extend([z_value] * len(frequency_array))

    # After all lines, set manual limits
    ax.set_ylim(min(all_frequency), max(all_frequency))
    # if phase_or_mag == MagnitudeOrPhase.Phase.value:
    ax.set_zlim(-200, 200)
    # else:
    #     ax.set_zlim(min(all_magnitude), max(all_magnitude))
    ax.set_xlim(min(all_times), max(all_times))

    ax.set_ylabel("Frequency (MHz)", labelpad=15)
    ax.set_zlabel(
        (
            "Magnitude (dB)"
            if data_frame_to_plot["mag_or_phase"].iloc[0]
            == MagnitudeOrPhase.Magnitude.value
            else "Phase (°)"
        ),
        labelpad=15,
    )
    ax.set_xlabel("Time (s)", labelpad=15)

    measured_value = data_frame_to_plot["mag_or_phase"].iloc[0]
    if measured_value == MagnitudeOrPhase.Magnitude.value:
        ax.set_title(f"|{s_parameter}| \n Gesture {gesture}", y=0.95)
    else:
        ax.set_title(f"Phase {s_parameter}\n Gesture {gesture}", y=0.95)

    ax.invert_xaxis()
    ax.set_box_aspect([8, 5, 3])
    ax.view_init(elev=20, azim=30, roll=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    if file_output_flag:
        if fname is None:
            fname = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}-{s_parameter}-{measured_value}-{gesture}-{c_map_name}.{filetype}"
        if file_output_path is None:
            file_output_path = get_graph_path()
        if experiment_label:
            file_output_path = os.path.join(file_output_path, experiment_label)
            os.makedirs(file_output_path, exist_ok=True)
        file_output_path = os.path.join(file_output_path, measured_value, s_parameter)
        os.makedirs(file_output_path, exist_ok=True)
        plt.savefig(
            os.path.join(file_output_path, fname), format=filetype, transparent=True
        )

    return fig


def scale_3d_plot(ax: Axes, x_scale=1, y_scale=1, z_scale=1) -> Axes:
    """
    Scaling is done from here...
    """
    x_scale = 10
    y_scale = 4.5
    z_scale = 4.5

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj
    return ax


def plot_3d_plots_for_all_gestures_for_sparam(
    results_df: pd.DataFrame,
    s_param: SParam,
    mag_or_phase: MagnitudeOrPhase,
    cmap=DEFAULT_COLOUR_MAP,
    save_to_file: bool = False,
    experiment_label=None,
    filetype=DEFAULT_FILE_TYPE,
) -> [Figure]:

    # df passing is by reference so make a copy
    results_df = results_df.copy(deep=True)
    figures: [Figure] = []
    results_df = convert_magnitude_rows_to_db(results_df)
    experiments = results_df["label"].unique()

    output_df = None
    for experiment in experiments:
        # get all the same label experiments -> this means the same gesture
        same_gesture = results_df[
            (results_df["label"] == experiment)
            & (results_df["s_parameter"] == s_param.value)
            & (results_df["mag_or_phase"] == mag_or_phase.value)
        ]

        single_gesture = same_gesture[
            same_gesture["id"] == random.choice(same_gesture["id"].unique())
        ]
        # output will contain one unique gesture capture for each
        output_df: pd.DataFrame = pd.concat(
            [output_df, single_gesture], ignore_index=True
        )

    for val, single_gesture_df in output_df.groupby("label"):

        single_gesture_df = single_gesture_df.reset_index(drop=True)
        chosen_gesture = single_gesture_df["label"][0].split("_")[-1]

        group = single_gesture_df.iloc[:, 4:].groupby("time")
        figures.append(
            plot_3d_time_series(
                single_gesture_df,
                c_map_name=cmap,
                file_output_flag=save_to_file,
                experiment_label=experiment_label,
                filetype=filetype,
            )
        )

    return figures


def plot_time_series(
    data_frame_to_plot,
    target_s_params,
    gestures_to_plot,
    experiment_label,
    target_frequency,
):

    data_frame_to_plot = convert_magnitude_rows_to_db(data_frame_to_plot)

    for target_s_param in target_s_params:
        plot_multiple_gestures_on_time_series(
            data_frame=data_frame_to_plot,
            experiment_label=experiment_label,
            gestures=gestures_to_plot,
            target_s_param=target_s_param,
            mag_or_phase=MagnitudeOrPhase.Magnitude,
            target_frequency=target_frequency,
        )


def get_s_param_data(results_df, s_param):
    return results_df[results_df["s_param"] == s_param]


def three_dplottest():

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(projection="3d")

    # Test data: simple 3D line
    x = [1, 2, 3]
    y = [4, 5, 6]
    z = [7, 8, 9]

    # Create segments (3 points)
    segments = np.array([[[1, 4, 7], [2, 5, 8]], [[2, 5, 8], [3, 6, 9]]])

    for i in range(10):
        # Create line collection (with color)
        line_collection = Line3DCollection(segments, colors="blue", linewidths=2)
        # Add the collection to the axis
        ax.add_collection3d(line_collection)
        segments += 1

    # Set limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    ax.set_zlim(6, 20)

    # Show plot
    plt.show()


if __name__ == "__main__":
    from matplotlib import colormaps

    data_capture_df = open_pickled_object(
        r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Patent_exp\Data Capture\17_09_patent_exp_combined_df.pkl"
    )
    data_capture_df.columns = list(data_capture_df.columns[:5]) + [
        int(val) for val in list(data_capture_df.columns[5:])
    ]
    s_param_test = [
        SParam.S11,
    ]

    # three_dplottest()
    for colour in list(colormaps):
        for param in s_param_test:
            plot_3d_plots_for_all_gestures_for_sparam(
                data_capture_df[
                    data_capture_df["label"] == "single_liquidAntennaSM3_A"
                ],
                param,
                MagnitudeOrPhase.Magnitude,
                cmap=colour,
                save_to_file=True,
                experiment_label="cmap_test",
                filetype="png",
            )

    # sns.set(rc={"xtick.bottom": True, "ytick.left": True}, font_scale=2)
    # pkl_classifier_folder = r"C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\classifiers\smd_3_patent_exp"
    #
    # full_df = open_pickled_object(
    #     r"C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\full_dfs\17_09_patent_exp_combined_df.pkl"
    # )
    #
    # confusion_matrix_option = ConfusionMatrixKey.FULL_SVM.value
    # results_df = open_pickled_object(
    #     r"C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\full_classification_results\smd_3_patent_exp.pkl"
    # )
    # accuracy_df = results_df[(results_df["gesture"] == "accuracy")]
    # accuracy_df = accuracy_df.sort_values(by="f1-score", ascending=False)
    # mag_df = accuracy_df[accuracy_df["type"] == "magnitude"]
    # top_magnitude = mag_df.iloc[0]
    # filtered_df_s_param = full_df[
    #     full_df["s_parameter"].isin(top_magnitude["s_param"].split("_"))
    # ]
    #
    # filtered_df_fq_range = filter_cols_between_fq_range(
    #     filtered_df_s_param,
    #     ghz_to_hz(float(top_magnitude["low_frequency"])),
    #     ghz_to_hz(float(top_magnitude["high_frequency"])),
    # )
    # confusion_matrix_dict = get_full_results_df_from_classifier_pkls(
    #     pkl_classifier_folder, extract="confusion_matrix"
    # )
    # display_confusion_matrix_for_top_n_values(
    #     full_df, results_df, confusion_matrix_dict
    # )

    # pickle_object(results_df, path=r'C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\full_classification_results', file_name='smd_3_patent_exp.pkl')
    #
    # #
    # # # replace experiment names for graphing
    # replace_dict = {
    #     "single_watchSmallAntennaL-140KHz-1001pts-10Mto4G": "Experiment 1",
    #     "single_flex-antenna-watch-140KHz-1001pts-10Mto4G": "Experiment 2",
    #     "filtered": "Filtered Features",
    #     "full": "Full Feature Set",
    #     "svm": "SVM",
    #     "dt": "Decision Tree"
    # }
    # results_df = results_df.replace(replace_dict)
    # results_df = results_df[(results_df['high_frequency'].astype(float) < 3.92)]
    # accuracy_df = results_df[(results_df["gesture"] == "accuracy") & (results_df['high_frequency'].astype(float) < 3.92)]

    # sampling_freq_results = gen_sweep_time_df()
    # plot_sampling_freq(sampling_freq_results)

    # Show plot
    # top_classifier_for_each_band(results_df, include_ALL_sparams=False)
    # max_accuracy_for_mag_sparam_categories(results_df)
    # freq_band_line_plot(results_df)
    # svm_vs_dt_strip_plot(results_df)
    # svm_vs_dtree_violin_plot(results_df)
    # full_vs_filtered_features_plot(results_df)
    # plot_s_param_mag_phase_from_touchstone(os.path.join(get_touchstones_path(), 'cp1_soil2_dry.s2p'), 'Watch Short Antenna')
    # plot_s_param_mag_phase_from_touchstone(os.path.join(get_touchstones_path(), 'watch_L_short_band_short_short_wires_140khz_1001pts.s2p'), 'Flex Antenna')

    # plt.show()
    # ax.legend(title='Classifier', labels=['SVM', 'D Tree'])
