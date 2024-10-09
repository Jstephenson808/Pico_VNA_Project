import os

from VNA_utils import pickle_object, open_pickled_object, get_label_from_pkl_path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from VNA_enums import DataFrameCols, SParam, DateFormats
from VNA_utils import get_pickle_path, get_classifiers_path, reorder_data_frame_columns
from VNA_data import VnaData

import pickle
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
from tsfresh import defaults

#todo class -> picoVNA converter
def pivot_data_frame_for_s_param(
    s_param: str, data_frame: pd.DataFrame, mag_or_phase: DataFrameCols
) -> pd.DataFrame:
    """
    Takes in a data_frame in DataFrameFormats format and returns a dataframe which
    has been pivoted to have the frequency as the column title with the other info
    (ie the s param, id, label, time) in seperate columns
    :param s_param: desired sparam for filtering
    :param data_frame: dataframe to be pivoted
    :param mag_or_phase: magnitude or phase selection for pivoting
    :return: pivoted dataframe with the columns reordered
    """
    if (mag_or_phase is not DataFrameCols.MAGNITUDE) and (
        mag_or_phase is not DataFrameCols.PHASE
    ):
        raise ValueError(
            f"mag_or_phase must be one of those, currently is {mag_or_phase}"
        )
    sparam_df = data_frame[data_frame[DataFrameCols.S_PARAMETER.value] == s_param]
    new_df = sparam_df.pivot(
        index=DataFrameCols.TIME.value,
        columns=DataFrameCols.FREQUENCY.value,
        values=mag_or_phase.value,
    )
    new_df.reset_index(inplace=True)
    new_df["mag_or_phase"] = mag_or_phase.value
    new_df[DataFrameCols.S_PARAMETER.value] = s_param
    new_df[DataFrameCols.ID.value] = data_frame[DataFrameCols.ID.value]
    new_df[DataFrameCols.LABEL.value] = data_frame[DataFrameCols.LABEL.value]
    reordered_columns = [
        DataFrameCols.ID.value,
        DataFrameCols.LABEL.value,
        "mag_or_phase",
        DataFrameCols.S_PARAMETER.value,
        DataFrameCols.TIME.value,
    ] + list(new_df.columns[1:-4])

    new_df = new_df[reordered_columns]
    return new_df

#todo move to vnaData -> this is a converter method
def csv_directory_to_ml_data_frame(directory: str) -> pd.DataFrame:
    """
    converts a given directory containing .csv data
    :param directory:
    :return:
    """
    csvs = os.listdir(directory)
    combined_data_frame = None
    for csv_fname in csvs:
        data = VnaData(os.path.join(directory, csv_fname))
        # loop over each sparam in the file and make a pivot table then append
        for sparam in data.data_frame[DataFrameCols.S_PARAMETER.value].unique():
            pivoted_data_frame = pivot_data_frame_for_s_param(
                sparam, data.data_frame, DataFrameCols.MAGNITUDE
            )
            combined_data_frame = pd.concat(
                (combined_data_frame, pivoted_data_frame), ignore_index=True
            )

            pivoted_data_frame = pivot_data_frame_for_s_param(
                sparam, data.data_frame, DataFrameCols.PHASE
            )
            combined_data_frame = pd.concat(
                (combined_data_frame, pivoted_data_frame), ignore_index=True
            )

    return combined_data_frame


def combine_dfs_with_labels(directory_list, labels) -> pd.DataFrame:
    ids = [i for i in range(len(directory_list))]
    new_df = csv_directory_to_ml_data_frame(directory_list.pop(0), labels.pop(0), ids.pop(0))
    for dir, label, sample_id in zip(directory_list, labels, ids):
        temp_df = csv_directory_to_ml_data_frame(dir, label, sample_id)
        new_df = pd.concat((new_df, temp_df), ignore_index=True)
    return new_df


def calulate_window_size_from_seconds(
    data_frame: pd.DataFrame, length_window_seconds: float
):
    return len(
        data_frame[(data_frame[DataFrameCols.TIME.value] < length_window_seconds)]
    )


def rolling_window_split(data_frame: pd.DataFrame, rolling_window_seconds: float):
    new_id_list = [i for i in range(100000)]
    new_df: pd.DataFrame = None
    movement_dict = {}

    # get each of the ids in turn
    grouped_data_id = data_frame.groupby([DataFrameCols.ID.value])

    # Iterate over the groups and store each filtered DataFrame
    for group_keys, group_data in grouped_data_id:
        window_size = rolling_window_seconds
        window_start = 0.0
        window_end = window_start + window_size
        # get the avg of a single set of time
        window_increment = (
            group_data[
                (group_data[DataFrameCols.S_PARAMETER.value] == SParam.S11.value)
                & (group_data["mag_or_phase"] == "magnitude")
            ][DataFrameCols.TIME.value]
            .diff()
            .mean()
        )
        while window_end <= group_data[DataFrameCols.TIME.value].max():
            new_id = new_id_list.pop(0)
            windowed_df = group_data[
                (group_data[DataFrameCols.TIME.value] >= window_start)
                & (group_data[DataFrameCols.TIME.value] < window_end)
            ]
            new_df, movement_dict = combine_windowed_df(
                new_df, windowed_df, new_id, movement_dict
            )

            window_start += window_increment
            window_end += window_increment

    # for measurement_id in ids:
    #     for mag_phase in data_frame["mag_or_phase"].unique():
    #         for s_param in data_frame[DataFrameCols.S_PARAMETER.value].unique():
    #             new_id = new_id_list.pop(0)
    #             # need to select each sparam in turn and then mag and phase in turn to make sure they are all with the same id
    #             id_frame = data_frame[(data_frame[DataFrameCols.ID.value] == measurement_id) & (data_frame[DataFrameCols.S_PARAMETER.value] == s_param) & (data_frame["mag_or_phase"] == mag_phase)]
    #             # number of indexes which map to that many seconds
    #             window_size = calulate_window_size_from_seconds(
    #                 id_frame, rolling_window_seconds
    #             )
    #             rolling_window = id_frame.rolling(window=window_size)
    #             for window_df in rolling_window:
    #                 if len(window_df) == window_size:
    #                     new_df, id_movement = combine_windowed_df(
    #                         new_df, window_df, new_id, id_movement
    #                     )
    return new_df, movement_dict


def window_split(data_frame: pd.DataFrame, window_seconds: float):
    new_id_list = [i for i in range(100000)]
    new_df: pd.DataFrame = None
    movement_dict = {}
    # get each of the ids in turn
    grouped_data_id = data_frame.groupby([DataFrameCols.ID.value])

    # Iterate over the groups and store each filtered DataFrame
    for group_keys, group_data in grouped_data_id:
        window_size = window_seconds
        window_start = 0.0
        window_end = window_start + window_size
        # get the avg of a single set of time
        window_increment = window_size
        while window_end <= group_data[DataFrameCols.TIME.value].max():
            new_id = new_id_list.pop(0)
            windowed_df = group_data[
                (group_data[DataFrameCols.TIME.value] >= window_start)
                & (group_data[DataFrameCols.TIME.value] < window_end)
            ]
            new_df, id_movement = combine_windowed_df(
                new_df, windowed_df, new_id, movement_dict
            )

            window_start += window_increment
            window_end += window_increment

    return new_df, movement_dict


def create_movement_vector_for_single_data_frame(df: pd.DataFrame) -> pd.Series:
    movement_dict = {}
    groups = df.groupby([DataFrameCols.ID.value])
    for id_value, id_df in groups:
        movement_dict[id_value[0]] = id_df[DataFrameCols.LABEL.value].values[0]
    return pd.Series(movement_dict)


def combine_windowed_df(
    new_df: pd.DataFrame, windowed_df: pd.DataFrame, new_id, movement_dict
) -> pd.DataFrame:
    windowed_df = windowed_df.reset_index(drop=True)

    windowed_df[DataFrameCols.ID.value] = new_id
    movement_dict[new_id] = windowed_df[DataFrameCols.LABEL.value][0]

    VnaData.zero_ref_time(windowed_df)
    if new_df is None:
        new_df = windowed_df
    else:
        new_df = pd.concat((new_df, windowed_df), ignore_index=True)
    return new_df, movement_dict


def split_data_frame_into_id_chunks(
    df: pd.DataFrame, ids_per_split: int
) -> [pd.DataFrame]:

    # Get the unique IDs
    unique_ids = df[DataFrameCols.ID.value].unique()

    # Initialize a list to store the smaller DataFrames
    split_dfs_by_id = []

    # Split into chunks of 3 IDs each
    for i in range(0, len(unique_ids), ids_per_split):
        # Get the current chunk of 3 IDs
        chunk_ids = unique_ids[i : i + ids_per_split]

        # Filter the original DataFrame for those IDs
        smaller_df = df[df[DataFrameCols.ID.value].isin(chunk_ids)]

        # Append the resulting DataFrame to the list
        split_dfs_by_id.append(smaller_df)

    return split_dfs_by_id


def extract_features_and_test(
    full_data_frame,
    feature_vector,
    drop_cols=[DataFrameCols.LABEL.value],
    n_jobs=defaults.N_PROCESSES,
    ids_per_split=0,
):
    combined_df = full_data_frame.ffill()
    # s_params_mapping = {s.value:index+1 for index, s in enumerate(SParam)}
    # full_data_frame[DataFrameCols.S_PARAMETER.value].map({s.value: index for index, s in enumerate(SParam)})
    data_frame_without_label = combined_df.drop(columns=drop_cols)
    if ids_per_split > 0:
        split_dfs = split_data_frame_into_id_chunks(
            data_frame_without_label, ids_per_split
        )
        features_list = [
            extract_features(
                df,
                column_sort=DataFrameCols.TIME.value,
                column_id=DataFrameCols.ID.value,
                n_jobs=n_jobs,
            )
            for df in split_dfs
        ]
        extracted = pd.concat(features_list)
    else:
        extracted = extract_features(
            data_frame_without_label,
            column_sort=DataFrameCols.TIME.value,
            column_id=DataFrameCols.ID.value,
            n_jobs=n_jobs,
        )
    extracted = impute(extracted)
    # print(extracted.head())
    # print(feature_vector.head())
    features_filtered = select_features(extracted, feature_vector)

    X_full_train, X_full_test, y_train, y_test = train_test_split(
        extracted, feature_vector, test_size=0.4
    )

    classifier_full = DecisionTreeClassifier()
    classifier_full.fit(X_full_train, y_train)
    classifier_full_y_pred = classifier_full.predict(X_full_test)
    decision_tree_full_dict = classification_report(
        y_test, classifier_full_y_pred, output_dict=True
    )
    decision_tree_full_confusion_matrix = confusion_matrix(
        y_test, classifier_full_y_pred
    )
    ConfusionMatrixDisplay(decision_tree_full_confusion_matrix).plot()
    print(classification_report(y_test, classifier_full_y_pred))

    X_filtered_train, X_filtered_test = (
        X_full_train[features_filtered.columns],
        X_full_test[features_filtered.columns],
    )
    dt_classifier_filtered = DecisionTreeClassifier()
    dt_classifier_filtered.fit(X_filtered_train, y_train)
    dt_classifier_filtered_y_pred = dt_classifier_filtered.predict(X_filtered_test)
    decision_tree_filtered_dict = classification_report(
        y_test, dt_classifier_filtered_y_pred, output_dict=True
    )
    decision_tree_filterd_confusion_matrix = confusion_matrix(
        y_test, dt_classifier_filtered_y_pred
    )
    print(classification_report(y_test, dt_classifier_filtered_y_pred))

    # print("SVM".center(80, "="))
    # Splitting the data into training and testing sets
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        extracted, feature_vector, test_size=0.4
    )

    # Standardizing the feature vectors
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)
    X_full_test_scaled = scaler.transform(X_full_test)

    # Creating an SVM classifier
    svm_classifier = SVC()

    # Training the SVM classifier
    svm_classifier.fit(X_full_train_scaled, y_train)
    # print("Full")
    # Evaluating the SVM classifier
    full_svm_y_pred = svm_classifier.predict(X_full_test_scaled)
    full_svm_confusion_matrix = confusion_matrix(y_test, full_svm_y_pred)
    full_svm_report = classification_report(y_test, full_svm_y_pred, output_dict=True)
    print(classification_report(y_test, full_svm_y_pred))

    # Splitting the data into training and testing sets
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        features_filtered, feature_vector, test_size=0.4
    )

    # Standardizing the feature vectors
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)
    X_full_test_scaled = scaler.transform(X_full_test)

    # Creating an SVM classifier
    svm_classifier_filtered = SVC()

    # Training the SVM classifier
    svm_classifier_filtered.fit(X_full_train_scaled, y_train)
    # print("Filtered")
    # Evaluating the SVM classifier
    svm_filtered_y_pred = svm_classifier_filtered.predict(X_full_test_scaled)
    dict_svm_filtered = classification_report(
        y_test, svm_filtered_y_pred, output_dict=True
    )
    svm_filtered_confusion_matrix = confusion_matrix(y_test, svm_filtered_y_pred)
    print(classification_report(y_test, svm_filtered_y_pred))

    # Evaluating the SVM classifier
    # print("Filtered")
    # Evaluating the SVM classifier

    return {
        "full_features": extracted,
        "filtered_features": features_filtered,
        "filtered_svm_report": dict_svm_filtered,
        "full_svm_report": full_svm_report,
        "full_dt_report": decision_tree_full_dict,
        "filtered_dt_report": decision_tree_filtered_dict,
        "filtered_dt_confusion_matrix": decision_tree_filterd_confusion_matrix,
        "full_dt_confusion_matrix": decision_tree_full_confusion_matrix,
        "full_svm_confusion_matrix": full_svm_confusion_matrix,
        "filtered_svm_confusion_matrix": svm_filtered_confusion_matrix,
    }


def test_features_print(full_features, features_filtered, feature_vector):
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        full_features, feature_vector, test_size=0.4
    )

    classifier_full = DecisionTreeClassifier()
    classifier_full.fit(X_full_train, y_train)
    print(classification_report(y_test, classifier_full.predict(X_full_test)))

    X_filtered_train, X_filtered_test = (
        X_full_train[features_filtered.columns],
        X_full_test[features_filtered.columns],
    )
    classifier_filtered = DecisionTreeClassifier()
    classifier_filtered.fit(X_filtered_train, y_train)
    print(classification_report(y_test, classifier_filtered.predict(X_filtered_test)))

    print("SVM".center(80, "="))
    # Splitting the data into training and testing sets
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        full_features, feature_vector, test_size=0.4
    )

    # Standardizing the feature vectors
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)
    X_full_test_scaled = scaler.transform(X_full_test)

    # Creating an SVM classifier
    svm_classifier = SVC()

    # Training the SVM classifier
    svm_classifier.fit(X_full_train_scaled, y_train)
    print("Full")
    # Evaluating the SVM classifier
    print(classification_report(y_test, svm_classifier.predict(X_full_test_scaled)))

    # Splitting the data into training and testing sets
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        features_filtered, feature_vector, test_size=0.4
    )

    # Standardizing the feature vectors
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)
    X_full_test_scaled = scaler.transform(X_full_test)

    # Creating an SVM classifier
    svm_classifier_filtered = SVC()

    # Training the SVM classifier
    svm_classifier_filtered.fit(X_full_train_scaled, y_train)
    print("Filtered")
    # Evaluating the SVM classifier
    print(
        classification_report(
            y_test, svm_classifier_filtered.predict(X_full_test_scaled)
        )
    )

    return {
        "filered_classifier": classifier_filtered,
        "full_classifier": classifier_full,
        "svm_full": svm_classifier,
        "svm_filtered": svm_classifier_filtered,
        "full_features": full_features,
        "filtered_features": features_filtered,
    }


def make_columns_have_s_param_mag_phase_titles(
    data_frame: pd.DataFrame,
) -> pd.DataFrame:
    freq_cols = [val for val in data_frame.columns.values if isinstance(val, int)]
    grouped_data = data_frame.groupby(["mag_or_phase", DataFrameCols.S_PARAMETER.value])
    new_combined_df = None
    for keys, df in grouped_data:
        label_to_add = ("_").join(keys)
        new_cols = [f"{label_to_add}_{col_title}" for col_title in freq_cols]
        df.rename(columns=dict(zip(freq_cols, new_cols)), inplace=True)
        df = df.drop(columns=[DataFrameCols.S_PARAMETER.value, "mag_or_phase"])
        if new_combined_df is None:
            new_combined_df = df
        else:
            new_combined_df = pd.merge(
                new_combined_df,
                df,
                on=[
                    DataFrameCols.ID.value,
                    DataFrameCols.TIME.value,
                    DataFrameCols.LABEL.value,
                ],
            )
    return new_combined_df


def filter_cols_between_fq_range(df: pd.DataFrame, lower_bound, upper_bound):
    """
    Filter the data frame so only the fq window of interest is selected and all teh
    :param df:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    freq_cols = get_list_of_in_bounds_fq(df, lower_bound, upper_bound)
    return filter_columns(df, freq_cols)


def get_list_of_in_bounds_fq(df, lower_bound_fq_hz, upper_bound_fq_hz):
    """
    Df has columns which correspond to fq, mixed in with labels for those measurements, need to remove non int
    columm labels and then return a list of fq points which are within bounds of the fq range in Hz
    :param df: data frame which contains
    :param lower_bound_fq_hz:
    :param upper_bound_fq_hz:
    :return: list of columns which are in fq range
    """
    cols = df.columns.values
    # Filter out non-integer values
    filtered_list = [x for x in cols if isinstance(x, int)]
    # Filter the list based on the provided bounds
    freq_cols = [
        x for x in filtered_list if lower_bound_fq_hz <= x <= upper_bound_fq_hz
    ]
    return freq_cols


def filter_columns(df, frequencies):
    """
    filtering of the data frame is done via regex, the filter function filters
    by keeping labels from axis (columns) for which re.search(regex, label) == True.
    :param df:
    :param frequencies:
    :return:
    """
    pattern = rf"^id$|^label$|^mag_or_phase$|^s_parameter$|^time$"
    if frequencies:
        pattern += "|" + "|".join(f"^{num}$" for num in frequencies)
    return df.filter(regex=pattern, axis=1)


def feature_extract_test_filtered_data_frame(
    filtered_data_frame,
    movement_vector,
    save=True,
    fname=None,
    n_jobs=defaults.N_PROCESSES,
):
    df_fixed = make_columns_have_s_param_mag_phase_titles(filtered_data_frame)
    classifiers = extract_features_and_test(df_fixed, movement_vector, n_jobs=n_jobs, ids_per_split=100)
    if save:
        if fname is None:
            fname = f"classifier_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}"
        else:
            fname = f"{fname}_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}"
        pickle_object(classifiers, path=get_classifiers_path(), file_name=fname)
    return classifiers, fname


def combine_data_frames_from_csv_folder(csv_folder_path, *, save=True, label=""):
    data_folders = os.listdir(csv_folder_path)
    combined_df: pd.DataFrame = None
    for data_folder in data_folders:
        combined_df_for_one_folder = csv_directory_to_ml_data_frame(
            os.path.join(csv_folder_path, data_folder)
        )
        combined_df = pd.concat(
            (combined_df, combined_df_for_one_folder), ignore_index=True
        )

    if save:
        full_df_path = os.path.join(get_pickle_path(), "full_dfs")
        if label == "":
            fname = f"full_combined_df_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}.pkl"
        else:
            fname = f"{label}_full_combined_df_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}.pkl"

        os.makedirs(full_df_path, exist_ok=True)
        with open(
            os.path.join(
                full_df_path,
                fname,
            ),
            "wb",
        ) as f:
            pickle.dump(combined_df, f)

    return combined_df


def extract_gesture_metric_values(
    classifier_dict: dict,
    report_keys: [str],
    *,
    gesture="weighted avg",
    metric="f1-score",
) -> dict:
    metric_list = []
    for report_key in report_keys:
        features_report = classifier_dict[report_key]
        gesture_report = features_report[gesture]
        """
        The F-beta score can be interpreted as a weighted harmonic mean of
        the precision and recall, where an F-beta score reaches its best
        value at 1 and worst score at 0.

        The F-beta score weights recall more than precision by a factor of
        ``beta``. ``beta == 1.0`` means recall and precision are equally important.
        """
        metric_value = gesture_report[metric]
        metric_list.append(metric_value)
    return metric_list


def extract_gesture_metric_to_df(
    pickle_fnames,
    *,
    gesture="weighted avg",
    metric="f1-score",
    folder_path=get_pickle_path(),
) -> pd.DataFrame:
    f1_scores = {}
    for fname in pickle_fnames:
        path = os.path.join(folder_path, fname)
        print(os.path.basename(path))
        classifier_dict = open_pickled_object(path)
        label = get_label_from_pkl_path(path)
        columns = [x for x in classifier_dict.keys() if "report" in x]
        f1_scores[label] = extract_gesture_metric_values(
            classifier_dict, columns, gesture=gesture, metric=metric
        )

    return pd.DataFrame.from_dict(f1_scores, orient="index", columns=columns)


def extract_full_results_to_df(
    pickle_fnames, folder_path=get_pickle_path(), extract="report"
) -> pd.DataFrame:
    full_results_data_frame = None
    for fname in pickle_fnames:
        path = os.path.join(folder_path, fname)
        print(os.path.basename(path))
        classifier_dict = open_pickled_object(path)
        label = get_label_from_pkl_path(path)
        report_keys = [x for x in classifier_dict.keys() if extract in x]
        if extract == "report":
            partial_results_df = extract_all_metrics_to_df(
                classifier_dict, report_keys, label
            )
        if extract == "features":
            partial_results_df = extract_feature_number(
                classifier_dict, report_keys, label
            )
        full_results_data_frame = pd.concat(
            (full_results_data_frame, partial_results_df), ignore_index=True
        )

    if extract == "report":
        full_results_data_frame = fix_measurement_column(full_results_data_frame)
    return full_results_data_frame


def extract_feature_number(
    classifier_dict: dict, report_keys: [str], label
) -> pd.DataFrame:
    combined_data_frame = None
    for report_key in report_keys:
        report_key_dict = {}
        report_key_dict["Experiment"] = label.split("-")[0]
        report_key_dict["Feature Set"] = report_key
        report_key_dict["Number of Features"] = len(classifier_dict[report_key].columns)
        report_key_data_frame = pd.DataFrame.from_dict([report_key_dict])
        combined_data_frame = pd.concat(
            (combined_data_frame, report_key_data_frame), ignore_index=True
        )
    return combined_data_frame


def extract_all_metrics_to_df(
    classifier_dict: dict, report_keys: [str], label
) -> pd.DataFrame:
    combined_data_frame = None
    for report_key in report_keys:
        report_key_data_frame = pd.DataFrame.from_dict(
            classifier_dict[report_key]
        ).T.reset_index()
        split_column = report_key_data_frame["index"].str.rsplit("_", n=1, expand=True)
        # need to fix these rows as they don't have correct label
        rows_to_copy = (
            (split_column[0] == "accuracy")
            | (split_column[0] == "macro avg")
            | (split_column[0] == "weighted avg")
        )
        # copy over
        split_column.loc[rows_to_copy, 1] = split_column.loc[rows_to_copy, 0]
        # relabel whole column to fix this
        split_column[0] = split_column[0][0]
        split_column.columns = ["label", "gesture"]
        report_key_data_frame = pd.concat(
            [report_key_data_frame.drop("index", axis=1), split_column], axis=1
        )

        report_key_data_frame["parameters"] = label
        report_key_data_frame["classifier"] = report_key.split("_")[1]
        report_key_data_frame["full or filtered"] = report_key.split("_")[0]
        new_column_order = (
            list(report_key_data_frame.columns)[4:]
            + list(report_key_data_frame.columns)[:4]
        )
        report_key_data_frame = report_key_data_frame[new_column_order]
        combined_data_frame = pd.concat(
            (combined_data_frame, report_key_data_frame), ignore_index=True
        )
    return combined_data_frame


def get_results_from_classifier_pkls(folder_path):
    fnames = os.listdir(folder_path)
    weighted_f1_score_df = extract_gesture_metric_to_df(
        fnames, gesture="weighted avg", metric="f1-score", folder_path=folder_path
    )
    stacked_df = weighted_f1_score_df.stack()
    return stacked_df.sort_values(ascending=False)


def fix_measurement_column(results_df: pd.DataFrame) -> pd.DataFrame:
    pattern = r"(S?\d?\d?_?S?\d?\d?\w+)_(\w+)_(\d+\.\d+)_(\d+\.\d+)"
    results_df[["s_param", "type", "low_frequency", "high_frequency"]] = (
        results_df["parameters"].str.extractall(pattern).reset_index(drop=True)
    )
    results_df.drop(columns=["parameters"], inplace=True)
    results_df = reorder_data_frame_columns(
        results_df, [0, 2, 3, 9, 8, 10, 11, 1, 4, 5, 6, 7]
    )
    return results_df


def get_full_results_df_from_classifier_pkls(folder_path, extract="report"):
    fnames = os.listdir(folder_path)

    return extract_full_results_to_df(fnames, folder_path, extract)


if __name__ == "__main__":
    pass

    # will open the first file in the combined df folder for testing
    # combined_df = open_full_results_df(os.listdir(get_combined_df_path())[0])

    # classifier_pickles = os.listdir(os.path.join(get_pickle_path(), "classifiers"))
    # classifiers = {fname.split('.')[0]:open_pickled_object(fname) for fname in classifier_pickles}
    # os.makedirs(get_pickle_path(), exist_ok=True)

    # rolling_df, rolling_movement = rolling_window_split(combined_df, 2.0)
    # rolling_movement_vector = pd.Series(rolling_movement.values())
    #
    # rolling_all_Sparams_magnitude = rolling_df[(rolling_df['mag_or_phase'] == "magnitude")]
    #
    # windowed_df, windowed_movement_dict = window_split(combined_df, 2.0)
    # windowed_movement_vector = pd.Series(windowed_movement_dict.values())
    #
    # windowed_all_Sparams_magnitude = windowed_df[(windowed_df['mag_or_phase'] == "magnitude")]
    # min_frequency, max_frequency = ghz_to_hz(5.81), ghz_to_hz(6)
    # low_frequency, high_frequency = min_frequency, min_frequency + mhz_to_hz(100)
    # while high_frequency <= max_frequency:
    #     print(f"{hz_to_ghz(low_frequency)}GHz->{hz_to_ghz(high_frequency)}GHz")
    #     rolling_all_Sparams_magnitude_filtered = filter_cols_between_fq_range(rolling_all_Sparams_magnitude,
    #                                                                           low_frequency,
    #                                                                           high_frequency)
    #     windowed_all_Sparams_magnitude_filtered = filter_cols_between_fq_range(windowed_all_Sparams_magnitude,low_frequency, high_frequency)
    #     result = feature_extract_test_filtered_data_frame(rolling_all_Sparams_magnitude_filtered, rolling_movement_vector, fname=f"rolling_all_Sparams_magnitude_{hz_to_ghz(low_frequency)}_{hz_to_ghz(high_frequency)}")
    #     result_2 = feature_extract_test_filtered_data_frame(windowed_all_Sparams_magnitude_filtered, windowed_movement_vector, fname=f"windowed_all_Sparams_magnitude_{hz_to_ghz(low_frequency)}_{hz_to_ghz(high_frequency)}")
    #     low_frequency += mhz_to_hz(100)
    #     high_frequency += mhz_to_hz(100)
