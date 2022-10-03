from pathlib import Path

import numpy as np
import pandas as pd
# import pickle

if __name__ == '__main__':
    # determine dataset folder and get column names
    dataset_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/dataset')
    submissions_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/')
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read evaluation_ids file ... ]")
    evaluation_path = str(dataset_path.joinpath('evaluation_ids.csv'))
    col_list = ['row_id']  # , 'cell_id'
    evaluation_ids = pd.read_csv(evaluation_path, usecols=col_list)
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read cite predictions ... ]")
    cite_sub = np.load(str(submissions_path.joinpath('cite', 'my', 'old_true.npy')), mmap_mode='r')
    # with open(str(submissions_path.joinpath('cite', 'kaggle', 'tune-lgbm-only-final.pickle')), 'rb') as f:
    #     cite_sub = pickle.load(f)
    cite_sub = cite_sub.flatten()
    evaluation_ids['target'] = pd.Series(cite_sub)
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read mutiome predictions ... ]")
    multiome_sub = str(submissions_path.joinpath('multiome', 'ura',
                                                 'mmsc_svd_gena_features_gene_atac_gene_id_all_29.09.csv'))
    multiome_sub = pd.read_csv(multiome_sub)
    evaluation_ids['target'].loc[cite_sub.shape[0]:] = multiome_sub['target'].loc[cite_sub.shape[0]:]
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Make submission ... ]")
    del evaluation_ids['row_id']
    evaluation_ids = evaluation_ids.dropna()
    evaluation_ids = evaluation_ids.reset_index(drop=True)
    assert not evaluation_ids['target'].isna().any()
    save_path = str(submissions_path.joinpath('both',
                                              'old_true-mmsc_svd_gena_features_gene_atac_gene_id_all_29.09.csv'))
    evaluation_ids.to_csv(save_path, index_label='row_id')
    print(f"[ Make submission is done. ]")
