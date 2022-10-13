import torch
from torch import nn

from common import get_bite_pos_emb


def get_inputs(vector: np.array):
    bin_mask = vector > 0
    inds = np.where(bin_mask)
    non_zero_vals = vector[inds]

    bins = []
    for x in non_zero_vals:
        bins.append(get_bite_pos_emb(int(x), pad_len=9, pad_side='left'))

    bins = np.vstack(bins)

    return inds[0], bins


class MostSimpleCellEncoder(nn.Module):
    def __init__(self,
                 feature_len: int,
                 emb_dim: int,
                 emb_normalization: bool = True,
                 val_emb_mode: str = 'sum'):
        super(MostSimpleCellEncoder, self).__init__()
        self.feature_len = feature_len
        self.pos_embeddings = nn.Embedding(feature_len, emb_dim, max_norm=emb_normalization)
        self.val_embeddings = nn.EmbeddingBag(feature_len, emb_dim, max_norm=emb_normalization, mode=val_emb_mode)

    def forward(self, value_bin_ind: torch.Tensor, bin_mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            value_bin_ind: матрица формата (bs, feature_len, bin_repr_len), содержащая индексы положений единичек
                           в бинарном представлении;
            bin_mask: матрица формата (bs, feature_len), состоящая из нулей и единиц;
                      Маппятся ненулевые позиции в ATAC векторе;

        Returns:
            тензор формата (bs, emb_dim) который содержит в себе эмбеддинг клетки
        """
        # Получаем матрицу эмбеддингов
        pos_indexes = torch.arange(self.feature_len)[None, ...]  # (1, feature_len)
        pos_indexes = pos_indexes.repeat(value_bin_ind.shape[0], 1)  # (bs, feature_len)
        pos_emb = self.pos_embeddings(pos_indexes)  # (bs, feature_len, emb_dim)
        # Получаем матрицу значений
        # todo: что делать с пустыми списками? | Возможно стоит использовать паддинг в эмбеддинг слое
        val_emb = self.val_embeddings(value_bin_ind)  # (bs, feature_len, emb_dim)
        # Вычисляем эмбеддинг клетки
        emb = torch.mean(((val_emb + pos_emb) * bin_mask), dim=1)  # (bs, emb_dim)

        return emb
