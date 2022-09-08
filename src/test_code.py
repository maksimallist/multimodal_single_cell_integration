from torch.optim.lr_scheduler import CyclicLR

scheduler = CyclicLR(optim, max_lr=(cfg['MAX_LR']))
