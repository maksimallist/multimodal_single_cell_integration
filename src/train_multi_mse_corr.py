from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, PearsonCorrCoef
from tqdm import tqdm

from src.dataset import FlowDataset
from src.model import EncoderConfig, MultiModel
from src.watchers import ExpWatcher


def corr_loss(predict: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> torch.Tensor:
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


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.min_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss) -> None:
        if loss > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == '__main__':
    root = Path(__file__).absolute().parent.parent
    exp_root = root.joinpath('experiments')
    watcher = ExpWatcher(exp_name='multi_mse_corr', root=exp_root)
    # data paths
    train_features_path = root.joinpath('dataset', 'train_multi_inputs.h5')
    train_targets_path = root.joinpath('dataset', 'train_multi_targets.h5')
    test_features_path = root.joinpath('dataset', 'test_multi_inputs.h5')

    # log paths
    watcher.log("train_dataset", train_features_path=str(train_features_path))
    watcher.log("train_dataset", train_targets_path=str(train_targets_path))

    # load dataset
    seed = watcher.rlog('train', random_seed=42)
    torch.manual_seed(seed)  # fix random generator state
    device = torch.device('cuda')
    watcher.log('train', device='cuda')
    print(f"[ Load the dataset from hard disk ... ]")
    train_dataset = FlowDataset(features_file=str(train_features_path), targets_file=str(train_targets_path),
                                transform=add_dimension, device=device)
    test_dataset = FlowDataset(features_file=str(test_features_path), transform=add_dimension, device=device)

    # create dataloaders
    batch_size = watcher.rlog('train', batch_size=32)
    shuffle_mode = watcher.rlog('train', shuffle_mode=True)
    print(f"[ Split train data on train and valid set. ]")
    train_dataset, valid_dataset = train_val_split(train_dataset, 0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_mode)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[ Load dataset is complete. ]")

    # model
    model_name = watcher.rlog('model', name='enc&reg_head')
    encoder_filters = watcher.rlog('model', encoder_filters=(1, 8, 8, 32, 128, 512, 128, 32, 8, 1))
    encoder_kernels = watcher.rlog('model', encoder_kernels=(15, 5, 5, 5, 5, 5, 5, 5, 3))
    enc_out = watcher.rlog('model', encoder_out=448)
    dec_in = watcher.rlog('model', decoder_in=484)
    channels = watcher.rlog('model', channels=1)

    encoder_conf = EncoderConfig(filters=encoder_filters, kernels=encoder_kernels)
    model = MultiModel(encoder_conf, enc_out=enc_out, dec_in=dec_in).to(device)
    p_num = 0
    for p in model.parameters():
        p_num += np.prod(np.array(p.shape))

    watcher.log('model', trainable_parameters=p_num)
    # watcher.writer.add_graph(model, next(iter(train_dataloader))[1])
    print(f"Number of trainable parameters in model: {p_num};")

    # Loss function and optimizer
    watcher.log('train', optimizer='torch.optim.Adam')
    learning_rate = watcher.rlog('train', lr=0.01)
    opt_betas = watcher.rlog('train', opt_betas=(0.5, 0.5))
    model_optimizer = Adam(model.parameters(), lr=learning_rate, betas=opt_betas)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='max', factor=0.1, patience=4,
                                  threshold=0.001, min_lr=0.000001, verbose=True)

    # Metrics
    mse = MeanSquaredError().to(device)
    p_corr = PearsonCorrCoef().to(device)

    # Train loop
    epochs = watcher.rlog('train', epohs=40)
    verbose_every = watcher.rlog('train', print_loss_every=20000)
    watcher.save_config()
    watcher.add_model_checkpoint_callback(mode='max', check_freq=1, verbose=1)

    mse_loss = torch.nn.MSELoss()

    print(f"[ Start training ... ]")
    model.train()
    batch_number = len(train_dataloader)
    step = 0
    for e in range(epochs):
        print(f"[ Training epoch: {e + 1} ------------------------------ ]")
        with tqdm(train_dataloader, miniters=verbose_every, desc='Batch', disable=False) as progress:
            for i, (cell_id, x, y) in enumerate(progress):
                step += i + (e + 1) * (batch_number // batch_size)

                # forward pass
                pred = torch.squeeze(model(x))

                mse_tensor = mse_loss(pred, y)
                corr_tensor = torch.mean(corr_loss(pred, y, normalize=True))
                loss = 0.5 * mse_tensor + 0.5 * corr_tensor

                # backward pass
                model_optimizer.zero_grad()
                loss.backward()

                # update weights
                model_optimizer.step()

                # calculate mse metric
                err = mse(pred, y)
                watcher.add_scalar('mse_error', err, step)

                # calculate Pearson correlation metric
                for v_x, v_y in zip(pred, y):
                    p_corr(v_x, v_y)
                corr = p_corr.compute()
                watcher.add_scalar('pearson_correlation', corr, step)

                # log loss
                mse_numpy = mse_tensor.detach().cpu().numpy()
                watcher.add_scalar('Loss-MSE', mse_numpy, step)

                corr_numpy = corr_tensor.detach().cpu().numpy()
                watcher.add_scalar('Loss-Correlation', corr_numpy, step)

                loss_numpy = loss.detach().cpu().numpy()
                watcher.add_scalar('Common Loss', loss_numpy, step)
                # if i % verbose_every == 0:
                #     print(f"[ Loss: {loss_n} | MSE: {err} | Pearson-corr: {corr} ]")

            err = mse.compute()
            mse.reset()
            corr = p_corr.compute()
            p_corr.reset()

        # -------------------------------------------------------------------

        # print(f"[ Epoch {e + 1} is complete. | MSE: {err} | Pearson-corr: {corr} ]")
        watcher.save_model(train_step=e, trainable_rule=model, name=model_name)

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
                watcher.add_scalar('Valid-Pearson_Correlation', corr, step)
                watcher.add_scalar('Valid-MSE', err, step)
                mse.reset()
                p_corr.reset()
        print(f"[ Validation is complete. | MSE: {err} | Pearson-corr: {corr} ]")
        scheduler.step(corr)
        watcher.is_model_better(monitor=corr,
                                step=step,
                                model=model,
                                optimizer=model_optimizer)
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
    pred_file_path = str(watcher.exp_root.joinpath('multi_eval.npy'))
    ids_path_file = str(watcher.exp_root.joinpath('multi_ids.npy'))
    np.save(pred_file_path, test_pred)
    np.save(pred_file_path, ids)
    print(f"[ Prediction on test set is complete. ]")
