from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def load_pickle(path):
    return pd.read_pickle(path)


def fast_load_dir(directory_path):
    paths = list(Path(directory_path).glob("*.pkl"))

    # Using ProcessPoolExecutor for parallel reading
    with ProcessPoolExecutor() as executor:
        # map returns results in the same order as the input paths
        results = list(tqdm(executor.map(load_pickle, paths), total=len(paths)))

    return pd.concat(results, ignore_index=True)
