from tsfresh import defaults

from VNA_enums import DataFrameCols


class FeatureExtractionParameters:

    def __init__(self,
                 n_jobs=defaults.N_PROCESSES,
                 ids_per_split=0,
                 drop_cols=None,
                 show_warnings: bool = defaults.SHOW_WARNINGS,
                 disable_extraction_progressbar: bool = defaults.DISABLE_PROGRESSBAR
                 ):
        # default arg can't be mutable
        if drop_cols is None:
            drop_cols = [DataFrameCols.LABEL.value]
        self.n_jobs = n_jobs
        self.drop_cols = drop_cols
        self.ids_per_split = ids_per_split
        self.show_warnings = show_warnings
        self.disable_extraction_progressbar = disable_extraction_progressbar
