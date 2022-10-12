from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, PearsonCorrCoef
from tqdm import tqdm

from src.common import pearson_corr_loss, train_val_split
from src.dataset import FSCCDataset
from src.models import BurtsevForwardModelConf, BurtsevForwardModel
from src.watchers import ExpWatcher


def f2img(x: np.array) -> np.array:
    l = np.sqrt(x.shape[0])
    if l % int(l) != 0:
        z = int(l)
        new_shape = (z, z + 1)
        new_x = np.pad(x, (0, ((z * (z + 1)) - x.shape[0])))
        new_x = np.reshape(new_x, (1, z, z + 1))
    else:
        z = int(l)
        new_x = np.reshape(x, (1, z, z))

    new_x = new_x / np.max(new_x)

    return new_x


def t2img(x: np.array) -> np.array:
    l = np.sqrt(x.shape[0])
    if l % int(l) != 0:
        z = int(l)
        new_shape = (z + 1, z + 1)
        new_x = np.pad(x, (0, ((z + 1) ** 2 - x.shape[0])))
        new_x = np.reshape(new_x, (1, z + 1, z + 1))
    else:
        z = int(l)
        new_x = np.reshape(x, (1, z, z))

    return new_x


if __name__ == '__main__':
    root = Path(__file__).absolute().parent.parent
    exp_root = root.joinpath('experiments')
    watcher = ExpWatcher(exp_name='multi_img_mse_corr', root=exp_root)
    # data paths
    dataset_path = root.joinpath('dataset')

    # load dataset
    seed = watcher.rlog('train', random_seed=42)
    torch.manual_seed(seed)  # fix random generator state
    device = torch.device('cuda')
    watcher.log('train', device='cuda')
    print(f"[ Load the dataset from hard disk ... ]")
    train_dataset = FSCCDataset(dataset_path, 'multi', 'train', transform=f2img, target_transform=t2img)
    test_dataset = FSCCDataset(dataset_path, 'multi', 'test', transform=f2img)

    # for test
    train_dataset.set_length(500)
    test_dataset.set_length(500)

    # remember shape for correct loss calculation
    true_y_shape = train_dataset.data_shapes['multi']['train']['targets'][1]

    # create dataloaders
    batch_size = watcher.rlog('train', batch_size=2)
    shuffle_mode = watcher.rlog('train', shuffle_mode=True)
    print(f"[ Split train data on train and valid set. ]")
    train_dataset, valid_dataset = train_val_split(train_dataset, 0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_mode)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[ Load dataset is complete. ]")

    # model
    model_name = watcher.rlog('model', name='forward_img_gene_model')
    model_config = BurtsevForwardModelConf((1, 478, 479), torch.nn.functional.leaky_relu)
    model = BurtsevForwardModel(model_config).to(device)
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
                                  threshold=0.0001, min_lr=0.000001, verbose=True)

    # Metrics
    mse = MeanSquaredError().to(device)
    p_corr = PearsonCorrCoef().to(device)

    # Train loop
    epochs = watcher.rlog('train', epohs=4)
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
            for i, meta_data in enumerate(progress):
                step += i + (e + 1) * (batch_number // batch_size)

                # prepare tensors
                # x = torch.from_numpy(meta_data['inputs']).to(device)
                # y = torch.from_numpy(meta_data['targets']).to(device)
                x = meta_data['inputs'].to(device)
                y = meta_data['targets'].to(device)

                # forward pass
                pred = model(x)

                # calc losses
                # print(y.shape, pred.shape)
                mse_tensor = mse_loss(pred, y)

                corr_y, corr_pred = y.flatten(start_dim=1)[:true_y_shape], pred.flatten(start_dim=1)[:true_y_shape]
                corr_tensor = torch.mean(pearson_corr_loss(corr_pred, corr_y, normalize=True))

                loss = 0.3 * mse_tensor + 0.7 * corr_tensor

                # backward pass
                model_optimizer.zero_grad()
                loss.backward()

                # normalize gradients
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad = param.grad / (torch.norm(param.grad) + 1e-8)

                # update weights
                model_optimizer.step()

                # calculate mse metric
                err = mse(pred, y)
                watcher.add_scalar('mse_error', err, step)

                # calculate Pearson correlation metric
                for v_x, v_y in zip(corr_pred, corr_y):
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
                for i, meta_data in enumerate(val_progress):
                    x = meta_data['inputs'].to(device)
                    y = meta_data['targets'].to(device)

                    pred = model(x)
                    corr_y, corr_pred = y.flatten(start_dim=1)[:true_y_shape], pred.flatten(start_dim=1)[:true_y_shape]
                    # calculate metric
                    err = mse(pred, y)
                    # calculate Pearson correlation metric
                    for v_x, v_y in zip(corr_pred, corr_y):
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
    test_pred = []
    with tqdm(test_dataloader, desc='Batch', disable=False) as progress:
        for i, meta_data in enumerate(progress):
            x = meta_data['inputs'].to(device)
            p = model(x).detach()
            p = torch.flatten(p, start_dim=1)[:true_y_shape]
            p = p.cpu().numpy()
            test_pred.append(p)

    test_pred = np.vstack(test_pred)
    pred_file_path = str(watcher.exp_root.joinpath('multi_eval.npy'))
    np.save(pred_file_path, test_pred)
    print(f"[ Prediction on test set is complete. ]")
