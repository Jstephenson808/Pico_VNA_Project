import random
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

from ml_model import (extract_full_results_to_df, get_results_from_classifier_pkls,
                      get_full_results_df_from_classifier_pkls,
                      filter_cols_between_fq_range)

matplotlib.use("TkAgg")
from VNA_utils import (
    get_full_results_df_path,
    reorder_data_frame_columns,
    get_touchstones_path,
    ghz_to_hz,
    get_frequency_column_headings_list, hz_to_ghz
)

from VNA_enums import ConfusionMatrixKey, DataFrameCols
from VNA_utils import pickle_object, open_pickled_object
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

def make_confusion_matrix_dict_string_from_series(series: pd.Series)->str:
    return ('_').join([series['s_param'], series['type'], series['low_frequency'], series['high_frequency']])

def confusion_matrix_from_single_result(single_result_series: pd.Series,
                                        labels,
                                        confusion_matrix_dict,
                                        confusion_matrix_option: ConfusionMatrixKey)->None:

    confusion_matrix_key = make_confusion_matrix_dict_string_from_series(single_result_series)
    confusion_matrix = confusion_matrix_dict[confusion_matrix_key][confusion_matrix_option]
    ConfusionMatrixDisplay(confusion_matrix, display_labels=labels).plot()
    plt.title(
        f'Confusion Matrix for {single_result_series["label"].split("_")[-1]} \n\r '
        f'Using {single_result_series["classifier"].title()} classifier between {single_result_series["low_frequency"]} and {single_result_series["high_frequency"]} GHz')
    plt.show()
    return

def display_confusion_matrix_for_top_value(full_df, results_df, confusion_matrix_dict):
    full_df.columns = [pd.to_numeric(col, errors='coerce') if col.isnumeric() else col for col in full_df.columns]

    accuracy_df = results_df[(results_df["gesture"] == "accuracy")]
    accuracy_df = accuracy_df.sort_values(by='f1-score', ascending=False)
    mag_df = accuracy_df[accuracy_df['type'] == 'magnitude']
    top_magnitude = mag_df.iloc[0]


    labels = [label[-1] for label in full_df['label'].unique()]
    confusion_matrix_from_single_result(top_magnitude, labels, confusion_matrix_dict, confusion_matrix_option)

def plot_fq_time_series(full_df: pd.DataFrame,*, s_parameter=None, mag_or_phase=None, label=None, n_random_ids=1, target_frequency=None):
    if target_frequency is None:
        raise AttributeError("No target frequency")

    if s_parameter is None or mag_or_phase is None or label is None:
        raise AttributeError(f"Must include all params s_param={s_parameter}, mag_or_phase={mag_or_phase}, label={label}")

    filtered_df = full_df.query(
        f's_parameter == "{s_parameter}" and mag_or_phase == "{mag_or_phase}" and label == "{label}"')
    grouped_by_id = filtered_df.groupby('id')
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
            filtered_df[closest_fq],
        )
    ax.set_ylabel(f"|{s_parameter}|")
    ax.set_xlabel("Time (s)")
    plt.title(f"|{s_parameter}| Over Time at {hz_to_ghz(closest_fq)} GHz")
    plt.show()


def filter_fq_cols(df, target_frequency):
    closest_fq_col = get_closest_freq_column(df, target_frequency)
    # this filters all teh columns
    # fq_cols = [int(col) for col in [col for col in test.columns.astype(str) if re.search(r'\d', col)]]
    str_cols = [col for col in df.columns if type(col) is str]
    str_cols.append(closest_fq_col)
    return df[str_cols]


def get_closest_freq_column(data_frame, target_frequency):
    fq_series = get_frequency_column_headings_list(data_frame)
    closest_fq_col = fq_series[(np.abs(np.asarray(fq_series) - target_frequency)).argmin()]
    return closest_fq_col

def plot_comparison_table(full_df, *, s_parameter=None, mag_or_phase=None, target_frequency=None):
    if target_frequency is None:
        raise AttributeError("No target frequency")

    if s_parameter is None or mag_or_phase is None:
        raise AttributeError(
            f"Must include all params s_param={s_parameter}, mag_or_phase={mag_or_phase}")

    filtered_df = full_df.query(
        f's_parameter == "{s_parameter}" and mag_or_phase == "{mag_or_phase}"')

    grouped_by_label = filtered_df.groupby('id')
    dfs_to_plot = []
    for label_group in grouped_by_label:

        grouped_by_ids = label_group.groupby('id')
        random_test = random.choice(list(grouped_by_ids.groups.keys()))
        dfs_to_plot.append(filter_fq_cols(filtered_df.query(f'id == "{random_test}"'), target_frequency))

    fig, axs = plt.subplots(nrows=len(dfs_to_plot), ncols=1, sharex=True)

    for ax, df in zip(axs, dfs_to_plot):
        closest_fq = get_closest_freq_column(df, target_frequency)
        ax.plot(df[DataFrameCols.TIME.value], df[closest_fq])
        ax.set_ylabel(f"|{s_parameter}|")
        ax.set_xlabel("Time (s)")
        plt.title(f"|{s_parameter}| Over Time at {hz_to_ghz(closest_fq)} GHz")
        plt.show()

def display_confusion_matrix_for_given_accuracy_value(full_df, results_df, confusion_matrix_dict, accuracy_value, measurement: MeasurementKey, confusion_matrix_option: ConfusionMatrixKey)->None:
    # allows you to index the lowest values by using -ve numbers
    if accuracy_value > 0:
        accuracy_index = accuracy_value - 1
    else:
        accuracy_index = accuracy_value

    full_df.columns = [pd.to_numeric(col, errors='coerce') if col.isnumeric() else col for col in full_df.columns]

    accuracy_df = results_df[(results_df["gesture"] == "accuracy")]
    accuracy_df = accuracy_df.sort_values(by='f1-score', ascending=False)
    measurement_df = accuracy_df[accuracy_df['type'] == measurement.value]
    selected_result = measurement_df.iloc[accuracy_index]

    labels = [label[-1] for label in full_df['label'].unique()]
    confusion_matrix_from_single_result(selected_result, labels, confusion_matrix_dict, confusion_matrix_option)

def show_top_five_and_bottom_five_confusion_matricies(full_df, results_df, confusion_matrix_dict, measurement: MeasurementKey, confusion_matrix_option: ConfusionMatrixKey)->None:
    for i in range(5):
        display_confusion_matrix_for_given_accuracy_value(full_df, results_df, confusion_matrix_dict, i, measurement, confusion_matrix_option)
    for i in range(-5,-1):
        display_confusion_matrix_for_given_accuracy_value(full_df, results_df, confusion_matrix_dict, i, measurement, confusion_matrix_option)

def plot_s_param_channels(full_df, target_freq_hz, measurement: MeasurementKey, s_params_to_plot=None):

    full_df = full_df[full_df['mag_or_phase'] == measurement.value]

    #find closest fq to target
    closest_fq = find_nearest_frequency(full_df, target_freq_hz)
    filter_id = full_df['id'].unique()[0]
    filtered_df = full_df[full_df['id'] == filter_id]
    if s_params_to_plot is None:
        s_params_to_plot = filtered_df['s_parameter'].unique()
    fig, ax = plt.subplots(len(s_params_to_plot), 1, sharex=True)
    ax = ax.flatten()
    for axis, s_param in zip(ax, s_params_to_plot):
        s_param_df = filtered_df[filtered_df['s_parameter'] == s_param]
        sns.lineplot(data=s_param_df, x='time', y=closest_fq, ax=axis)
        axis.set_title(f'{s_param}')
    plt.show()

def find_nearest_frequency(full_df, target_frequency_hz):
    array = np.asarray(list(full_df.columns)[5:])
    idx = (np.abs(array - target_frequency_hz)).argmin()
    return array[idx]

if __name__ == "__main__":
    sns.set(rc={"xtick.bottom": True, "ytick.left": True}, font_scale=2)
    pkl_classifier_folder = r'C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\classifiers\smd_3_patent_exp'

    full_df = open_pickled_object(r'C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\full_dfs\17_09_patent_exp_combined_df.pkl')

    confusion_matrix_option = ConfusionMatrixKey.FULL_SVM.value
    results_df = open_pickled_object(r'C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\full_classification_results\smd_3_patent_exp.pkl')
    accuracy_df = results_df[(results_df["gesture"] == "accuracy")]
    accuracy_df = accuracy_df.sort_values(by='f1-score', ascending=False)
    mag_df = accuracy_df[accuracy_df['type'] == 'magnitude']
    top_magnitude = mag_df.iloc[0]
    filtered_df_s_param = full_df[full_df['s_parameter'].isin(top_magnitude['s_param'].split('_'))]

    filtered_df_fq_range = filter_cols_between_fq_range(filtered_df_s_param,
                                                        ghz_to_hz(float(top_magnitude['low_frequency'])),
                                                        ghz_to_hz(float(top_magnitude['high_frequency'])))
    confusion_matrix_dict = get_full_results_df_from_classifier_pkls(
        pkl_classifier_folder,
        extract="confusion_matrix")
    display_confusion_matrix_for_top_value(full_df, results_df, confusion_matrix_dict)

    #pickle_object(results_df, path=r'C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\full_classification_results', file_name='smd_3_patent_exp.pkl')
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

    #Show plot
    # top_classifier_for_each_band(results_df, include_ALL_sparams=False)
    # max_accuracy_for_mag_sparam_categories(results_df)
    # freq_band_line_plot(results_df)
    # svm_vs_dt_strip_plot(results_df)
    # svm_vs_dtree_violin_plot(results_df)
    # full_vs_filtered_features_plot(results_df)
    #plot_s_param_mag_phase_from_touchstone(os.path.join(get_touchstones_path(), 'cp1_soil2_dry.s2p'), 'Watch Short Antenna')
    #plot_s_param_mag_phase_from_touchstone(os.path.join(get_touchstones_path(), 'watch_L_short_band_short_short_wires_140khz_1001pts.s2p'), 'Flex Antenna')

    #plt.show()
    # ax.legend(title='Classifier', labels=['SVM', 'D Tree'])
