import random
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("TkAgg")
from VNA_utils import (
    get_full_results_df_path,
    reorder_data_frame_columns,
    get_touchstones_path, get_classifier_path,
)
from ml_model import open_pickled_object, pickle_object
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from skrf.io import touchstone
from skrf import plotting

sns.set_theme(style="whitegrid")
# sns.set(font_scale=2)


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

    #top_n_groups = melted.groupby("type")["value"].max().nlargest(n_to_plot).index

    melted_df = melted.sort_values(['value'], ascending=False)[:n_to_plot]
    melted_df['type'] = melted_df['type'].map(lambda x:x.replace('Magnitude', 'Mag'))
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


def best_parameter_measurement_violin(results_df, n_to_plot=6, include_all=False):
    accuracy_df = results_df[results_df["gesture"] == "accuracy"]
    accuracy_df.loc[:, "type"] = accuracy_df["type"] + "_" + accuracy_df["s_param"]
    melted = melt_and_filter_mag_sparam(accuracy_df, include_all)
    top_n_groups = melted.groupby("type")["value"].mean().nlargest(n_to_plot).index

    melted_df = melted[melted["type"].isin(top_n_groups)]
    fig, ax = plt.subplots()
    sns.boxplot(data=melted_df, x="type", y="value", hue="label")
    # sns.move_legend(
    #     ax, loc="lower right", ncol=2, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
    #     labels=['SVM', 'D Tree']
    # )
    ax.set(
        xlabel="Experiment",
        ylabel="Classifier Accuracy",
        title=f"SVM vs Decision Tree Classification Accuracy \n Showing The Top {n_to_plot} S Parameter Combinations By Mean Accuracy",
    )

    legend = plt.legend(loc="lower right")


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


if __name__ == "__main__":
    sns.set(rc={"xtick.bottom": True, "ytick.left": True}, font_scale=2)
    #results_df = open_pickled_object(os.path.join(get_full_results_df_path(), "watch_small_ant.pkl"))

    results_df = open_pickled_object(os.path.join(get_classifier_path(), "all_Sparams_magnitude_1.0_1.1_2024_08_09.pkl"))

    print(results_df['full_dt_confusion_matrix'])
    disp = ConfusionMatrixDisplay(confusion_matrix=results_df['full_dt_confusion_matrix'])
    disp.plot()

    #
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
    # # accuracy_df = results_df[(results_df["gesture"] == "accuracy") & (results_df['high_frequency'].astype(float) < 3.92)]
    #
    # # sampling_freq_results = gen_sweep_time_df()
    # # plot_sampling_freq(sampling_freq_results)
    #
    # #Show plot
    # # top_classifier_for_each_band(results_df, include_ALL_sparams=False)
    #max_accuracy_for_mag_sparam_categories(results_df)
    # freq_band_line_plot(results_df)
    # svm_vs_dt_strip_plot(results_df)
    # svm_vs_dtree_violin_plot(results_df)
    # full_vs_filtered_features_plot(results_df)
    #plot_s_param_mag_phase_from_touchstone(os.path.join(get_touchstones_path(), 'cp1_soil2_dry.s2p'), 'Watch Short Antenna')
    #plot_s_param_mag_phase_from_touchstone(os.path.join(get_touchstones_path(), 'watch_L_short_band_short_short_wires_140khz_1001pts.s2p'), 'Flex Antenna')

    plt.show()
    # ax.legend(title='Classifier', labels=['SVM', 'D Tree'])
