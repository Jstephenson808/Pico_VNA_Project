import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from VNA_enums import DataFrameCols, SParam, DateFormats
from VNA_utils import get_data_path, get_pickle_path, get_classifier_path
from VNA_data import VnaData

import pickle
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
from tsfresh import defaults


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


def make_fq_df(directory: str) -> pd.DataFrame:
    csvs = os.listdir(os.path.join(get_data_path(), directory))
    combined_data_frame = None
    for csv_fname in csvs:
        data = VnaData(os.path.join(get_data_path(), directory, csv_fname))
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
    new_df = make_fq_df(directory_list.pop(0), labels.pop(0), ids.pop(0))
    for dir, label, sample_id in zip(directory_list, labels, ids):
        temp_df = make_fq_df(dir, label, sample_id)
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

def create_movement_vector_for_single_data_frame(df: pd.DataFrame)-> pd.Series:
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


def extract_features_and_test(
        full_data_frame, feature_vector, drop_cols=[DataFrameCols.LABEL.value], n_jobs=defaults.N_PROCESSES
):
    combined_df = full_data_frame.ffill()
    # s_params_mapping = {s.value:index+1 for index, s in enumerate(SParam)}
    # full_data_frame[DataFrameCols.S_PARAMETER.value].map({s.value: index for index, s in enumerate(SParam)})
    dropped_label = combined_df.drop(columns=drop_cols)
    extracted = extract_features(
        dropped_label,
        column_sort=DataFrameCols.TIME.value,
        column_id=DataFrameCols.ID.value,
        n_jobs=n_jobs
    )
    impute(extracted)
    features_filtered = select_features(extracted, feature_vector)

    X_full_train, X_full_test, y_train, y_test = train_test_split(
        extracted, feature_vector, test_size=0.4
    )

    classifier_full = DecisionTreeClassifier()
    classifier_full.fit(X_full_train, y_train)
    decision_tree_full_dict = classification_report(
        y_test, classifier_full.predict(X_full_test), output_dict=True
    )
    print(classification_report(y_test, classifier_full.predict(X_full_test)))

    X_filtered_train, X_filtered_test = (
        X_full_train[features_filtered.columns],
        X_full_test[features_filtered.columns],
    )
    classifier_filtered = DecisionTreeClassifier()
    classifier_filtered.fit(X_filtered_train, y_train)
    decision_tree_filtered_dict = classification_report(
        y_test, classifier_filtered.predict(X_filtered_test), output_dict=True
    )
    print(classification_report(y_test, classifier_filtered.predict(X_filtered_test)))

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
    full_svm_report = classification_report(
        y_test, svm_classifier.predict(X_full_test_scaled), output_dict=True
    )
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
    # print("Filtered")
    # Evaluating the SVM classifier
    dict_svm_filtered = classification_report(
        y_test, svm_classifier_filtered.predict(X_full_test_scaled), output_dict=True
    )
    print(
        classification_report(
            y_test, svm_classifier_filtered.predict(X_full_test_scaled)
        )
    )

    # Evaluating the SVM classifier
    # print("Filtered")
    # Evaluating the SVM classifier

    return {
        "filered_classifier": classifier_filtered,
        "full_classifier": classifier_full,
        "svm_full": svm_classifier,
        "svm_filtered": svm_classifier_filtered,
        "full_features": extracted,
        "filtered_features": features_filtered,
        "filtered_svm_report": dict_svm_filtered,
        "full_svm_report": full_svm_report,
        "full_dt_report": decision_tree_full_dict,
        "filtered_dt_report": decision_tree_filtered_dict,
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
    cols = df.columns.values
    # Filter out non-integer values
    filtered_list = [x for x in cols if isinstance(x, int)]
    # Filter the list based on the provided bounds
    freq_cols = [x for x in filtered_list if lower_bound <= x <= upper_bound]
    return filter_columns(df, freq_cols)


def filter_columns(df, frequencies):
    pattern = rf"^id$|^label$|^mag_or_phase$|^s_parameter$|^time$"
    if frequencies:
        pattern += "|" + "|".join(f"^{num}$" for num in frequencies)
    return df.filter(regex=pattern, axis=1)


def pickle_object(object_to_pickle,*, path:str, file_name:str):
    os.makedirs(path, exist_ok=True)
    if ".pkl" not in file_name[-4:]:
        file_name = f"{file_name}.pkl"
    path = os.path.join(path, file_name)
    with open(path, "wb") as f:
        pickle.dump(object_to_pickle, f)


def open_pickled_object(path):
    with open(path, "rb") as f:
        unpickled = pickle.load(f)
    return unpickled


def feature_extract_test_filtered_data_frame(
        filtered_data_frame, movement_vector, save=True, fname=None, n_jobs=defaults.N_PROCESSES
):
    df_fixed = make_columns_have_s_param_mag_phase_titles(filtered_data_frame)
    classifiers = extract_features_and_test(df_fixed, movement_vector, n_jobs=n_jobs)
    if save:
        if fname is None:
            fname = f"classifier_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}"
        else:
            fname = f"{fname}_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}"
        pickle_object(classifiers, path=get_classifiers_path(), file_name=fname)
    return classifiers, fname


def get_classifiers_path():
    return os.path.join(get_pickle_path(), "classifiers")


def get_full_dfs_path():
    return os.path.join(get_pickle_path(), "full_dfs")


def combine_data_frames_from_csv_folder(csv_folder_path, *, save=True, label=""):
    data_folders = os.listdir(csv_folder_path)
    combined_df: pd.DataFrame = None
    for data_folder in data_folders:
        combined_df_for_one_folder = make_fq_df(data_folder)
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


def get_label_from_pkl_path(path):
    """
    removes .pkl and then date from fname format
    "all_Sparams_magnitude_0.01_0.11_2024_04_02.pkl"
    """
    return os.path.basename(path)[::-1].split("_", maxsplit=3)[-1][::-1]


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
        pickle_fnames, *, gesture="weighted avg", metric="f1-score", folder_path=get_pickle_path()
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

def get_results_from_classifier_pkls(folder_path):
    fnames = os.listdir(folder_path)
    weighted_f1_score_df = extract_gesture_metric_to_df(
        fnames, gesture="weighted avg", metric="f1-score", folder_path=folder_path
    )
    stacked_df = weighted_f1_score_df.stack()
    return stacked_df.sort_values(ascending=False)


if __name__ == "__main__":
    pass

    # combined_df = open_pickled_object(
    #     os.path.join(get_pickle_path(), "full_dfs", os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]))

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
