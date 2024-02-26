import numpy as np
import numba

__all__ = [
    "one_hot_encode",
    "one_hot_decode",
]


@numba.njit
def one_hot_encode(data: np.ndarray):
    data = data.transpose(1, 2, 0)
    assert data.ndim == 3
    n = data.shape[2]

    assert n <= 30  # ensure int32 does not overflow
    # assert data.dtype == np.uint8
    for x in np.unique(data):
        assert x in [0, 1]

    # shift = np.arange(n, np.int32)[None, None]
    shift = np.zeros((1, 1, n), np.int32)
    shift[0, 0, :] = np.arange(0, n, 1, np.int32)

    binary = (data > 0)  # bool
    # after shift, numpy keeps int32, numba change dtype to int64
    binary = (binary << shift).sum(-1)  # move bit to left and combine to one
    binary = binary.astype(np.int32)

    return binary


@numba.njit
def one_hot_decode(data: np.ndarray, n: int):
    """
    returns (h, w, n) np.int64 {0, 1}
    """
    # shift = np.arange(n, dtype=np.int32)[None, None]
    shift = np.zeros((1, 1, n), np.int32)
    shift[0, 0, :] = np.arange(0, n, 1, np.int32)

    # x = np.array(data)[..., None]
    x = np.zeros((*data.shape, 1), data.dtype)
    x[..., 0] = data
    # after shift, numpy keeps int32, numba changes dtype to int64
    x = (x >> shift) & 1  # only keep the lowest bit, for each n

    x = x.transpose(2, 0, 1)
    return x
