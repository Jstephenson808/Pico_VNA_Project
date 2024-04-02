from ml_model import *
from VNA_utils import get_data_path

if __name__ == '__main__':
    combined_df = combine_data_frames_from_csv_folder(get_data_path(), label="single-watch-large-ant")
