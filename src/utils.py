import numpy as np


def get_bite_pos_emb(digit: int, pad_len: int, pad_side: str = 'right') -> np.array:
    sep = ','
    byte_string = np.base_repr(digit, base=2)
    byte_string = sep.join([byte_string[i] for i in range(len(byte_string))])
    byte_array = np.fromstring(byte_string, dtype=int, sep=sep)
    if byte_array.shape[0] < pad_len:
        if pad_side == 'right':
            byte_array = np.pad(byte_array, (0, pad_len - byte_array.shape[0]), 'constant', constant_values=0)
        elif pad_side == 'left':
            byte_array = np.pad(byte_array, (pad_len - byte_array.shape[0], 0), 'constant', constant_values=0)
        else:
            raise ValueError(f"The argument 'pad_side' can only take values from a list: ['right', 'left'], "
                             f"but {pad_side} was found.")

    return byte_array
