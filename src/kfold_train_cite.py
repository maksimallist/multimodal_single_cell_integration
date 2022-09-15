from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, PearsonCorrCoef
from tqdm import tqdm

from src.dataset import FlowDataset
from src.model import EncoderConfig, CiteModel
from src.watchers import ExpWatcher


def corr_error(predict: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    predict = predict - torch.mean(predict, dim=1).unsqueeze(1)
    target = target - torch.mean(target, dim=1).unsqueeze(1)
    loss_tensor = -torch.sum(predict * target, dim=1) / (target.shape[-1] - 1)  # minus because we want gradient ascend

    if normalize:
        s1 = torch.sqrt(torch.sum(predict * predict, dim=1) / (predict.shape[-1] - 1))
        s2 = torch.sqrt(torch.sum(target * target, dim=1) / (target.shape[-1] - 1))
        loss_tensor = loss_tensor / s1 / s2

    return loss_tensor


def add_dimension(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.unsqueeze(tensor, dim=dim)


def kfold_split(dataset, folds: int):
    fold_sizes = [len(dataset) // folds] * (folds - 1) + [len(dataset) // folds + len(dataset) % folds]
    ds_folds = torch.utils.data.random_split(dataset, fold_sizes, generator=torch.Generator().manual_seed(42))
    for fold in range(folds):
        yield torch.utils.data.ConcatDataset(ds_folds[:fold] + ds_folds[fold + 1:]), ds_folds[fold]


def train_val_split(dataset, val_volume: float = 0.2):
    train_examples = len(dataset)
    val_num = int(train_examples * val_volume)
    folds = torch.utils.data.random_split(dataset, [train_examples - val_num, val_num], generator=torch.Generator())
    return folds[0], folds[1]


if __name__ == '__main__':
    root = Path(__file__).absolute().parent.parent
    exp_name = Path(__file__).absolute().parent.name
    # data paths
    train_features_path = root.parent.joinpath('dataset', 'train_cite_inputs.h5')
    train_targets_path = root.parent.joinpath('dataset', 'train_cite_targets.h5')
    test_features_path = root.joinpath('dataset', 'test_cite_inputs.h5')

    print(f"[ Load the dataset from hard disk ... ]")
    seed = 42
    torch.manual_seed(seed)  # fix random generator state
    device = torch.device('cuda')
    train_dataset = FlowDataset(features_file=str(train_features_path), targets_file=str(train_targets_path),
                                transform=add_dimension, device=device)
    test_dataset = FlowDataset(features_file=str(test_features_path), transform=add_dimension, device=device)

    print(f"[ Start K-Flod split training ... ]")
    # ----------------------------------------------------------------------------
    for fold_ind, (ds_train, ds_eval) in enumerate(kfold_split(train_dataset, 5)):
        print(f"[ Train flod {fold_ind + 1}  ... ]")
        watcher = ExpWatcher(exp_name=f"fold_{fold_ind + 1}", root=root.joinpath(exp_name))
        watcher.log('train', random_seed=seed)
        watcher.log('train', device='cuda')

        # create dataloaders
        batch_size = watcher.rlog('train', batch_size=4)
        shuffle_mode = watcher.rlog('train', shuffle_mode=True)
        train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle_mode)
        valid_dataloader = DataLoader(ds_eval, batch_size=batch_size, shuffle=shuffle_mode)
        test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        print(f"[ Load dataset is complete. ]")

        # model
        model_name = watcher.rlog('model', name='enc&reg_head')
        encoder_filters = watcher.rlog('model', encoder_filters=(1, 32, 64, 128, 64, 32, 1))
        encoder_kernels = watcher.rlog('model', encoder_kernels=(15, 5, 5, 5, 5, 3))
        enc_out = watcher.rlog('model', encoder_out=345)
        encoder_conf = EncoderConfig(filters=encoder_filters, kernels=encoder_kernels)
        model = CiteModel(encoder_conf, enc_out=enc_out).to(device)
        p_num = 0
        for p in model.parameters():
            p_num += np.prod(np.array(p.shape))

        watcher.log('model', trainable_parameters=p_num)
        # watcher.writer.add_graph(model, next(iter(train_dataloader))[1])
        print(f"Number of trainable parameters in model: {p_num};")

        # Loss function and optimizer
        watcher.log('train', optimizer='torch.optim.Adam')
        learning_rate = watcher.rlog('train', lr=0.002)
        opt_betas = watcher.rlog('train', opt_betas=(0.5, 0.5))
        model_optimizer = Adam(model.parameters(), lr=learning_rate, betas=opt_betas)
        scheduler = ExponentialLR(model_optimizer, gamma=0.3)

        # Metrics
        mse = MeanSquaredError().to(device)
        p_corr = PearsonCorrCoef().to(device)

        # Train loop
        epochs = watcher.rlog('train', epohs=10)
        verbose_every = watcher.rlog('train', print_loss_every=200)
        watcher.save_config()
        watcher.add_model_checkpoint_callback(mode='max', check_freq=2, verbose=1)

        print(f"[ Start training ... ]")
        model.train()
        batch_number = len(train_dataloader)
        for e in range(epochs):
            print(f"[ Training epoch: {e + 1} ------------------------------ ]")
            description = f'Fold: {fold_ind} | Epoch: {e + 1} | Batch'
            with tqdm(train_dataloader, miniters=verbose_every, desc=description, disable=False) as progress:
                for i, (cell_id, x, y) in enumerate(progress):
                    step = i + (e + 1) * (batch_number // batch_size)
                    # forward pass
                    pred = torch.squeeze(model(x))
                    loss = corr_error(pred, y, normalize=True)
                    loss = torch.mean(loss)
                    # backward pass
                    model_optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    model_optimizer.step()
                    watcher.add_scalar('learning_rate', scheduler.get_last_lr()[0], step)

                    # calculate mse metric
                    err = mse(pred, y)
                    watcher.add_scalar('mse_error', err, step)

                    # calculate Pearson correlation metric
                    for v_x, v_y in zip(pred, y):
                        p_corr(v_x, v_y)
                    corr = p_corr.compute()
                    watcher.add_scalar('pearson_correlation', corr, step)

                    # log loss
                    loss_n = loss.detach().cpu().numpy()
                    watcher.add_scalar('Loss-Pearson_Correlation', loss_n, step)
                    if i % verbose_every == 0:
                        print(f"[ Loss: {loss_n} | MSE: {err} | Pearson-corr: {corr} ]")

                err = mse.compute()
                mse.reset()
                corr = p_corr.compute()
                p_corr.reset()

            # -------------------------------------------------------------------

            print(f"[ Epoch {e + 1} is complete. | MSE: {err} | Pearson-corr: {corr} ]")
            watcher.save_model(train_step=e, trainable_rule=model, name=model_name)
            scheduler.step()
            print()

            # -------------------------------------------------------------------
            print(f"[ Validation is started ... ]")
            with torch.no_grad():
                with tqdm(valid_dataloader, miniters=verbose_every, desc='Batch', disable=False) as val_progress:
                    for i, (cell_id, x, y) in enumerate(val_progress):
                        pred = torch.squeeze(model(x))
                        # calculate metric
                        err = mse(pred, y)
                        # calculate Pearson correlation metric
                        for v_x, v_y in zip(pred, y):
                            p_corr(v_x, v_y)
                        corr = p_corr.compute()

                    err = mse.compute()
                    corr = p_corr.compute().detach().cpu().numpy()
                    watcher.is_model_better(monitor=corr,
                                            step=step,
                                            model=model,
                                            optimizer=model_optimizer)

                    mse.reset()
                    p_corr.reset()
            print(f"[ Validation is complete. | MSE: {err} | Pearson-corr: {corr} ]")
        # -------------------------------------------------------------------
        watcher.writer.close()

        # -------------------------------------------------------------------
        print(f"[ Prediction on test set is started ... ]")
        # load weights of the best model
        weights_path = watcher.checkpoints_folder.joinpath('best_model', 'best_model.pt')
        model.load_state_dict(torch.load(weights_path)['model_state_dict'])
        model.to(device)
        model.eval()

        # start predictions on test set
        test_pred, ids = [], []
        with tqdm(test_dataloader, desc='Batch', disable=False) as progress:
            for i, (cell_ids, features) in enumerate(progress):
                ids.extend(cell_ids)
                p = model(features).detach()
                p = torch.flatten(p, start_dim=1)
                p = p.cpu().numpy()
                test_pred.append(p)

        test_pred = np.vstack(test_pred)
        pred_file_path = str(watcher.exp_root.joinpath('cite_eval.npy'))
        ids_path_file = str(watcher.exp_root.joinpath('cite_ids.npy'))
        np.save(ids_path_file, ids)
        np.save(pred_file_path, test_pred)
        print(f"[ Prediction on test set is complete. ]")
