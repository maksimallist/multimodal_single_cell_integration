import gc
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # determine dataset folder and get column names
    data_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/dataset')
    submissions_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/')

    # determine experiments folder
    exp_root = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/experiments/')

    # Make cite predictions
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read cite predictions ... ]")
    test_pred = np.load(str(exp_root.joinpath('check_fix_c-15.09.2022-14.12', 'cite_eval.npy')), mmap_mode='r')
    test_pred = test_pred.flatten()
    with open(str(submissions_path.joinpath('test.csv')), 'w') as f:
        f.write("row_id,target\n")
        for i, x in enumerate(tqdm(test_pred)):
            f.write(','.join([str(i), str(x)]) + '\n')

    del [test_pred]
    gc.collect()
    print(f"[ Reading cite predictions complete. ]")

    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read mutiome predictions ... ]")
    test_pred = np.load(str(exp_root.joinpath('check_fix_m-14.09.2022-20.16', 'multi_eval.npy')), mmap_mode='r')
    test_pred = test_pred.flatten()
    with open(str(submissions_path.joinpath('test.csv')), 'a') as f:
        for i, x in enumerate(tqdm(test_pred)):
            f.write(','.join([str(i), str(x)]) + '\n')

    del [test_pred]
    gc.collect()
    print(f"[ Reading mutiome predictions complete. ]")
