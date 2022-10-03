from pathlib import Path

import numpy as np
import pandas as pd
# import pickle

if __name__ == '__main__':
    # determine dataset folder and get column names
    submissions_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/')
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read mutiome predictions ... ]")
    multiome_sub = str(submissions_path.joinpath('multiome', 'kaggle', 'w-sparce-m-tsvd-ridge_notebook.csv'))
    multiome_sub = pd.read_csv(multiome_sub, index_col='row_id')
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read cite predictions ... ]")
    cite_sub = np.load(str(submissions_path.joinpath('cite', 'my', 'conv_mse_corr.npy')), mmap_mode='r')

    # with open(str(submissions_path.joinpath('cite', 'kaggle', 'tune-lgbm-only-final.pickle')), 'rb') as f:
    #     cite_sub = pickle.load(f)

    cite_sub = cite_sub.flatten()

    print(f"[ Make submission ... ]")
    multiome_sub['target'].loc[:cite_sub.shape[0] - 1] = cite_sub
    # assert not multiome_sub.isna().any()
    save_path = str(submissions_path.joinpath('both',
                                              'conv_mse_corr-w-sparce-m-tsvd-ridge_notebook.csv'))
    multiome_sub.to_csv(save_path)
    print(f"[ Make submission is done. ]")
