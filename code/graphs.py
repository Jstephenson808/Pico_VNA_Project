import pandas as pd

from VNA_utils import get_full_results_df_path
from ml_model import open_pickled_object
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    results_df = open_pickled_object(os.path.join(get_full_results_df_path(), "watch_L_ant_2.pkl"))
    results2 = results_df.copy()
    results2['label'] = 'test'
    results_df = pd.concat((results_df, results2))

    replace_dict = {'single_watchLargeAntennaL': 'Experiment 1', 'test': 'Experiment 2'}
    results_df = results_df.replace(replace_dict)

    svm_vs_dt_strip_plot(results_df)
    svm_vs_dtree_violin_plot(results_df)


    #ax.legend(title='Classifier', labels=['SVM', 'D Tree'])
    #svm_vs_dt_strip_plot(results_df, replace_dict)
    plt.show()