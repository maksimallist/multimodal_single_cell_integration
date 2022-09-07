from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from src.dataset import MyDataset
from model import EncoderConfig, Model
from watchers import ExpWatcher


def corr_error(predict: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    predict = predict - torch.mean(predict, dim=1).unsqueeze(1)
    target = target - torch.mean(target, dim=1).unsqueeze(1)
    loss_tensor = -torch.sum(predict * target, dim=1) / (target.shape[-1] - 1)  # minus because we want gradient ascend

    if normalize:
        s1 = torch.sqrt(torch.sum(predict * predict, dim=1) / (predict.shape[-1] - 1))
        s2 = torch.sqrt(torch.sum(target * target, dim=1) / (target.shape[-1] - 1))
        loss_tensor = loss_tensor / s1 / s2

    return loss_tensor


if __name__ == '__main__':
    root = Path(__file__).parent
    exp_root = root.joinpath('experiments').absolute()
    watcher = ExpWatcher(exp_name='debug_train_script', root=exp_root)
    # data paths
    meta_file = root.joinpath('dataset', 'metadata.csv')
    train_features_path = root.joinpath('dataset', 'train_multi_inputs.h5')
    train_targets_path = root.joinpath('dataset', 'train_multi_targets.h5')
    test_features_path = root.joinpath('dataset', 'test_multi_inputs.h5')

    # log paths
    watcher.log("train_dataset", train_features_path=str(train_features_path))
    watcher.log("train_dataset", train_targets_path=str(train_targets_path))
    watcher.log("test_dataset", test_features_path=str(test_features_path))

    # load dataset
    start = watcher.rlog("train_dataset", start_pos=0)
    end = watcher.rlog("train_dataset", end_pos=1000)
    print(f"[ Load a {end - start} lines of dataset from hard disk ... ]")
    train_dataset = MyDataset(features_file=str(train_features_path), start_pos=start, load_pos=end,
                              targets_file=str(train_targets_path),
                              transform=torch.from_numpy, target_transform=torch.from_numpy)

    test_dataset = MyDataset(features_file=str(test_features_path), start_pos=start, load_pos=end,
                             transform=torch.from_numpy)
    # create dataloaders
    batch_size = watcher.rlog('train', batch_size=2)
    shuffle_mode = watcher.rlog('train', shuffle_mode=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_mode)
    print(f"[ Load a {end - start} lines of dataset from hard disk is complete. ]")

    # model
    device = watcher.rlog('train', device='cuda')
    model_name = watcher.rlog('model', name='enc&reg_head')
    encoder_filters = watcher.rlog('model', encoder_filters=(1, 8, 8, 32, 128, 512, 128, 32, 8, 1))
    encoder_kernels = watcher.rlog('model', encoder_kernels=(15, 5, 5, 5, 5, 5, 5, 5, 3))
    enc_out = watcher.rlog('model', encoder_out=448)
    dec_in = watcher.rlog('model', decoder_in=484)
    channels = watcher.rlog('model', channels=1)

    encoder_conf = EncoderConfig(filters=encoder_filters, kernels=encoder_kernels)
    model = Model(encoder_conf, enc_out=enc_out, dec_in=dec_in).to(device)
    p_num = 0
    for p in model.parameters():
        p_num += np.prod(np.array(p.shape))

    watcher.log('model', trainable_parameters=p_num)
    print(f"Number of trainable parameters in model: {p_num};")

    # Loss function and optimizer
    watcher.log('train', optimizer='torch.optim.Adam')
    learning_rate = watcher.rlog('train', lr=0.002)
    opt_betas = watcher.rlog('train', opt_betas=(0.5, 0.5))
    model_optimizer = Adam(model.parameters(), lr=learning_rate, betas=opt_betas)

    # Metrics
    mse = MeanSquaredError().to(device)
    p_corr = PearsonCorrCoef().to(device)

    # Train loop
    epochs = watcher.rlog('train', epohs=2)
    watcher.save_config()
    print(f"[ Start training ... ]")
    for e in range(epochs):
        print(f"[ Training epoch: {e} ------------------------------ ]")
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            # forward pass
            x = x[:, None, :]
            pred = torch.squeeze(model(x))
            loss = corr_error(pred, y, normalize=True)
            loss = torch.mean(loss)
            # backward pass
            model_optimizer.zero_grad()
            loss.backward()
            # update weights
            model_optimizer.step()

            # calculate mse metric
            err = mse(pred, y)
            watcher.add_scalar('mse_error', err, i)

            # calculate Pearson correlation metric
            for v_x, v_y in zip(pred, y):
                p_corr(v_x, v_y)
            corr = p_corr.compute()
            watcher.add_scalar('pearson_correlation', corr, i)

            # log loss
            loss_n = loss.detach().cpu().numpy()
            watcher.add_scalar('Loss-Pearson_Correlation', loss_n, i)

            if i % 10 == 0:
                print(f"[ Loss value: {loss_n} | MSE metric: {err} | Pearson correlation coefficient: {corr} ]")

        err = mse.compute()
        mse.reset()
        corr = p_corr.compute()
        p_corr.reset()

        # -------------------------------------------------------------------
        print(f"[ Epoch number {e} is complete. MSE metric = {err} | Pearson correlation coefficient = {corr} ]")
        watcher.save_model(train_step=e, trainable_rule=model, name=model_name)
        print()
        # -------------------------------------------------------------------
        # todo: залогировать граф сети, а также сделать снимок архитектуры при помощи torchviz
        # todo: добавить кроссвалидацию и K-fold разбиение
        # todo: добавить шедулер на оптимизатор
        # todo: добавить подбор Lr
        # todo: добавить сохранение при улучшении метрики n раз (может есть общий callback для torch)
        # todo: зафиксировать seed торча для воспроизводимости результатов
        # todo: добавить код формирования файла для сабмита на тестовой части датасета

        # print(f"[ Validation is started ... ]")
        # with torch.no_grad():
        #     for i, (x, y) in enumerate(test_dataloader):
        #         x, y = x.to(device), y.to(device)
        #         x = x[:, None, :]
        #         pred = torch.squeeze(model(x))
        #         # calculate metric
        #         err = mse(pred, y)
        #         # corr = p_corr(pred, y)
        #
        #     err = mse.compute()
        #     mse.reset()
        #     # corr = p_corr.compute()
        #     # p_corr.reset()
        #     print(f"[ Validation is complete. MSE metric = {err} | Pearson correlation coefficient = XXX ]")
