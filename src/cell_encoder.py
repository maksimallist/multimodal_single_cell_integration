import torch
from torch import nn


class MostSimpleCellEncoder(nn.Module):
    def __init__(self,
                 feature_len: int,
                 emb_dim: int,
                 bin_len: int,
                 emb_normalization: bool = True,
                 val_emb_mode: str = 'sum'):
        super(MostSimpleCellEncoder, self).__init__()
        self.feature_len = feature_len
        self.bin_len = bin_len
        self.pos_embeddings = nn.Embedding(feature_len, emb_dim, max_norm=emb_normalization)
        self.val_embeddings = nn.EmbeddingBag(feature_len, emb_dim, max_norm=emb_normalization, mode=val_emb_mode)

    def binary(self, x: torch.Tensor):
        mask = 2 ** torch.arange(self.bin_len)  # .to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    def get_bin_repr(self, vals: torch.Tensor) -> torch.Tensor:
        binary_repr = torch.empty((*vals.shape, self.bin_len))
        for i, v in enumerate(vals):
            for j, q in enumerate(v):
                binary_repr[i][j] = self.binary(q)

        return binary_repr

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input_tensor: матрица формата (bs, feature_len, bin_repr_len), содержащая индексы положений единичек
                           в бинарном представлении;

        Returns:
            тензор формата (bs, emb_dim) который содержит в себе эмбеддинг клетки
        """
        nonzero_inds = input_tensor.nonzero()
        nonzero_vals = input_tensor[nonzero_inds]
        binary_repr = self.get_bin_repr(nonzero_vals)
        # Получаем матрицу эмбеддингов
        pos_emb = self.pos_embeddings(nonzero_inds)  # (bs, some_dim, emb_dim)
        # Получаем матрицу значений
        val_emb = self.val_embeddings(binary_repr)  # (bs, some_dim, emb_dim)
        # Вычисляем эмбеддинг клетки
        emb = torch.mean((val_emb + pos_emb), dim=1)  # (bs, emb_dim)

        return emb
