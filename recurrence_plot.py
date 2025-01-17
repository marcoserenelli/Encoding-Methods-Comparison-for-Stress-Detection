import enum
from typing import Union
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from fastdtw import fastdtw


def fast_e_dtw(t0, t1):
    distance, _ = fastdtw(t0, t1, dist=eucl_dist)
    return distance


def plot_recurrence_plot(data, name='', verbose=0):
    if verbose >= 1:
        print(data)
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='binary', origin='lower')
    plt.title('Recurrence Plot ' + name)
    plt.xlabel('Time')
    plt.ylabel('Time')
    plt.colorbar(label='Recurrence')
    return plt


def eucl_dist(x, y):
    dist = np.linalg.norm(x - y)
    return dist


def eucl_dist_traj(t1, t2):
    t1 = np.atleast_2d(t1)
    t2 = np.atleast_2d(t2)
    mdist = cdist(t1, t2, 'euclidean')
    return mdist


def e_dtw(t0, t1):
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = eucl_dist(t0[i - 1], t1[j - 1]) + min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
    dtw = C[n0, n1]
    return dtw


def e_edr(t0, t1, eps=0.5):
    n0 = len(t0)
    n1 = len(t1)
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            subcost = 0 if eucl_dist(t0[i - 1], t1[j - 1]) < eps else 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max(n0, n1)
    return edr


def e_erp(t0, t1, g=0.5):
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))

    gt0_dist = [abs(eucl_dist(g, x)) for x in t0]
    gt1_dist = [abs(eucl_dist(g, x)) for x in t1]
    mdist = eucl_dist_traj(t0, t1)

    C[1:, 0] = sum(gt0_dist)
    C[0, 1:] = sum(gt1_dist)
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            derp0 = C[i - 1, j] + gt0_dist[i - 1]
            derp1 = C[i, j - 1] + gt1_dist[j - 1]
            derp01 = C[i - 1, j - 1] + mdist[i - 1, j - 1]
            C[i, j] = min(derp0, derp1, derp01)
    erp = C[n0, n1]
    return erp


def reshape_ts(t: np.ndarray):
    assert isinstance(t, np.ndarray), "The time series must be a numpy array"
    t /= np.max(np.abs(t), axis=0)
    return t


class DistanceMethodName(enum.Enum):
    EUCLIDEAN = (eucl_dist, 'Euclidean')
    DTW = (fast_e_dtw, 'DTW')
    ERP = (e_erp, 'ERP')
    EDR = (e_edr, 'EDR')

    def __str__(self):
        return self.value[1]

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)


def create_multivariate_recurrence_plot(dataset: Union[list, np.ndarray], threshold: float = 0.7,
                                        distance: str = 'euclidean', verbose: int = 0, image_size: int = 128):
    """
    Given the dataset computes all the distances between all the timeseries producing a recurrence plot matrix.

    Args:
        dataset (Union[list, np.ndarray]): The input dataset containing time series data.
        threshold (float): The threshold for the distance measure.
        distance (str): The distance measure to use ('euclidean', 'dtw', 'erp', 'edr').
        verbose (int): The verbosity level.
        image_size (int): The size of the output image.

    Returns:
        np.ndarray: The resized recurrence plot matrix.
    """
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)
    assert len(dataset.shape) == 2, "The dataset must be a 2D array"

    n = dataset.shape[0]
    rp = np.zeros((n, n))
    dataset = reshape_ts(dataset)

    for i, t in enumerate(dataset):
        for j, s in enumerate(dataset):
            if i == j:
                rp[i][j] = 0
                continue
            if verbose >= 2:
                print(f'Checking ts {i} with {j}')
            if distance == 'euclidean':
                distance_value = eucl_dist_traj(t, s)
            elif distance == 'dtw':
                distance_value = fast_e_dtw(t, s)
            elif distance == 'erp':
                distance_value = e_erp(t, s)
            elif distance == 'edr':
                distance_value = e_edr(t, s)
            else:
                raise ValueError("Invalid distance measure. Choose from 'euclidean', 'dtw', 'erp', or 'edr'")
            t_dv = np.where(distance_value <= threshold, 1, 0)
            rp[i][j] = t_dv
        if verbose >= 1:
            print(rp)

    zoom_factor = (image_size / n, image_size / n)
    rp_resized = zoom(rp, zoom_factor, order=0)

    return rp_resized
