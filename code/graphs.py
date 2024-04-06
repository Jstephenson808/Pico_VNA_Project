import numpy as np
import pandas as pd

from VNA_utils import get_full_results_df_path, reorder_data_frame_columns
from ml_model import open_pickled_object, pickle_object
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sns.set_theme(style="whitegrid")
#sns.set(font_scale=2)


def svm_vs_dt_strip_plot(results_df: pd.DataFrame):
    accuracy_df = results_df[results_df['gesture'] == 'accuracy']

    melted = pd.melt(accuracy_df, id_vars=['label', 'classifier'], value_vars=['precision'])

    fig, ax = plt.subplots()
    sns.despine(bottom=True, left=True)
    sns.stripplot(data=melted, x='label', y='value', hue='classifier', dodge=True, alpha=.25, zorder=1, legend=False)

    sns.pointplot(
        data=melted, x='label', y='value', hue='classifier',
        dodge=.8 - .8 / 2, palette="dark", errorbar=None,
        markers="d", markersize=6, linestyle="none",
    )
    sns.move_legend(
        ax, loc="best", ncol=3, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
        labels=['SVM', 'Decision Tree']
    )
    ax.set(xlabel='Experiment', ylabel='Classifier Accuracy', title="SVM vs Decision Tree Classification Accuracy \n For Each Experiment")

    return

def svm_vs_dtree_violin_plot(results_df: pd.DataFrame):
    accuracy_df = results_df[results_df['gesture'] == 'accuracy']
    melted = pd.melt(accuracy_df, id_vars=['label', 'classifier'], value_vars=['precision'])

    fig, ax = plt.subplots()
    sns.violinplot(data=melted, x="label", y="value", hue="classifier")
    # sns.move_legend(
    #     ax, loc="lower right", ncol=2, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
    #     labels=['SVM', 'D Tree']
    # )
    ax.set(xlabel='Experiment', ylabel='Classifier Accuracy', title="SVM vs Decision Tree Classification Accuracy \n For Each Experiment")

    legend = plt.legend()

    # Access the legend object
    legend.set_title('Classifier')  # Set legend title

    # Set labels
    for text, label in zip(legend.get_texts(), ['SVM', 'D Tree']):
        text.set_text(label)
    return

def full_vs_filtered_features_plot(results_df:pd.DataFrame):
    accuracy_df = results_df[results_df['gesture'] == 'accuracy']
    melted = pd.melt(accuracy_df, id_vars=['full or filtered'], value_vars=['f1-score'])
    fig, ax = plt.subplots()
    sns.violinplot(data=melted, y="full or filtered", x="value", hue="full or filtered", legend=False)
    ax.set(xlabel='Classifier Accuracy', ylabel='Feature Set',
           title="Full vs Filtered Features Classification Accuracy")

def freq_band_line_plot(results_df:pd.DataFrame):
    accuracy_df = results_df[results_df['gesture'] == 'accuracy']
    melted = pd.melt(accuracy_df, id_vars=['label', 'low_frequency', 'high_frequency'], value_vars=['precision'])
    melted['mid_freq'] = round((melted['high_frequency'].astype(float) - 0.05), 2)
    fig, ax = plt.subplots()
    # Create a stripplot
    sns.lineplot(data=melted, x='mid_freq', y='value', hue='label',style="label", markers=True, dashes=False, sort=False)
    ax.set(ylabel='Classifier Accuracy', xlabel='Frequency Bands (GHz)',
           title='Classifier Accuracy For Each Tested Frequency Band')
    # ax.xaxis.set_major_locator(mticker.AutoMajorLocator(13))

    major_ticks = np.arange(0, 6, 0.5)  # Set major ticks every 2 units
    plt.xlim(0, 6)
    plt.xticks(major_ticks)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", bottom=True)
    #plt.gca().xaxis.set_minor_locator(mticker.AutoMinorLocator())
    # Set plot title

def fix_uscore_title_case(value):
    return (' ').join(value.split('_')).title()

def max_accuracy_for_mag_sparam_categories(results_df:pd.DataFrame, n_to_plot=6, include_all=False):
    accuracy_df = results_df[results_df['gesture'] == 'accuracy']
    melted = melt_and_filter_mag_sparam(accuracy_df, include_all)

    top_n_groups = melted.groupby('type')['value'].max().nlargest(n_to_plot).index

    melted_df = melted[melted['type'].isin(top_n_groups)]
    fig, ax = plt.subplots()
    sns.despine(bottom=True, left=True)
    sns.stripplot(data=melted_df, x="type", y="value", hue='label', dodge=True, alpha=.25, zorder=1, legend=False)
    # sns.move_legend(
    #     ax, loc="lower right", ncol=2, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
    #     labels=['SVM', 'D Tree']
    # )
    ax.set(xlabel='Experiment', ylabel='Classifier Accuracy',
           title="SVM vs Decision Tree Classification Accuracy \n For Each Experiment")

    legend = plt.legend(loc='lower right')

    # Show plot
    plt.show()


def melt_and_filter_mag_sparam(accuracy_df, include_all):
    # combine mag or phase and sparam
    accuracy_df.loc[:, 'type'] = accuracy_df['type'] + '_' + accuracy_df['s_param']
    melted = pd.melt(accuracy_df, id_vars=['label', 'type'], value_vars=['precision'])
    # melted[["type", "s_param"]].apply(tuple, axis=1)
    if not include_all:
        # inverting returned boolean df to remove these ~ is NOT
        # removes the "all" category
        melted = melted[~melted['type'].isin(['magnitude_all_Sparams', 'phase_all_Sparams'])]
    # fix the titles of the graph
    melted.loc[:, 'type'] = melted['type'].apply(fix_uscore_title_case)
    return melted


def best_parameter_measurement_violin(results_df, n_to_plot=6, include_all=False):
    accuracy_df = results_df[results_df['gesture'] == 'accuracy']
    accuracy_df.loc[:, 'type'] = accuracy_df['type'] + '_' + accuracy_df['s_param']
    melted = melt_and_filter_mag_sparam(accuracy_df, include_all)
    top_n_groups = melted.groupby('type')['value'].mean().nlargest(n_to_plot).index

    melted_df = melted[melted['type'].isin(top_n_groups)]
    fig, ax = plt.subplots()
    sns.violinplot(data=melted_df, x="type", y="value", hue='label')
    # sns.move_legend(
    #     ax, loc="lower right", ncol=2, frameon=True, columnspacing=1, handletextpad=0, title="Classifier",
    #     labels=['SVM', 'D Tree']
    # )
    ax.set(xlabel='Experiment', ylabel='Classifier Accuracy',
           title="SVM vs Decision Tree Classification Accuracy \n For Each Experiment")

    legend = plt.legend(loc='lower right')

    # Show plot
    plt.show()

if __name__ == '__main__':
   # sns.set(rc={"xtick.bottom": True, "ytick.left": True})
    results_df = open_pickled_object(os.path.join(get_full_results_df_path(), "watch_L_ant_2.pkl"))

    # just adding an extra experiment for testing
    results2 = results_df.copy()
    results2['label'] = 'test'
    results_df = pd.concat((results_df, results2), ignore_index=True)

    # replace experiment names for graphing
    replace_dict = {'single_watchLargeAntennaL': 'Experiment 1', 'test': 'Experiment 2', 'filtered':'Filtered Features', 'full':'Full Feature Set'}
    results_df = results_df.replace(replace_dict)

    max_accuracy_for_mag_sparam_categories(results_df)

    # Show plot
    plt.show()
    #freq_band_line_plot(results_df)
    # svm_vs_dt_strip_plot(results_df)
    # svm_vs_dtree_violin_plot(results_df)
    # full_vs_filtered_features_plot(results_df)

    #ax.legend(title='Classifier', labels=['SVM', 'D Tree'])
    #svm_vs_dt_strip_plot(results_df, replace_dict)
    #plt.show()