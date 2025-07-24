from pathlib import Path

from vna.VNA_utils import open_pickled_object

folder = Path(r'C:\Users\2573758S\Desktop\save_data')
pickles = folder.glob('*.pkl')
dfs = {pickle_path.stem: open_pickled_object(pickle_path) for pickle_path in pickles}