import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.svm import LinearSVC
import pandas as pd


def get_separating_plane(s1, s2, use_bias=True):
    # w = 4 * (s1 - s2) / np.linalg.norm(s1 - s2)
    # b = -(np.dot(s1, s1) + np.dot(s2, s2)) / (np.dot(s2, s2) - np.dot(s1, s1)) if use_bias else None
    unit = fit_regression(np.array([s1, s2]), [1, -1])

    return unit


def fit_svm(X, y, C=1e10, loss='hinge', penalty='l2', use_bias=True):
    try:
        X = X.detach().numpy()
    except AttributeError:
        pass
    clf = LinearSVC(C=C, loss=loss, penalty=penalty, max_iter=10000000, fit_intercept=use_bias)
    clf.fit(X, y.squeeze())

    print(f'Supports {np.where((2 * y - 1) * clf.decision_function(X) <= 1)[0]}')
    unit = get_unit(clf.coef_, bias=clf.intercept_ if use_bias else None)
    return unit


def fit_regression(X, y, use_bias=True):
    # reg = LinearRegression(fit_intercept=use_bias)
    reg = Ridge(alpha=0, solver='svd')
    reg.fit(X, y)

    unit = get_unit(reg.coef_, bias=reg.intercept_ if use_bias else None)
    return unit


def fit_partition(partition, X, mode='svm', use_bias=True):
    if mode == 'svm':
        y_partition = np.zeros(len(X))
        y_partition[(np.array(list(partition)))] = 1
        model = fit_svm(X, y_partition, use_bias=use_bias)
    elif mode == 'regression':
        y_partition = np.ones(len(X)) * -1
        y_partition[(np.array(list(partition)) - 1)] = np.arange(len(partition)) + 1
        model = fit_regression(X, y_partition, use_bias=use_bias)
    else:
        raise ValueError(f'Unknown mode {mode}')

    return model


def get_unit(weight, bias=None):
    unit = torch.nn.Linear(in_features=len(weight), out_features=1, bias=bias is not None)
    unit.weight.data = torch.tensor(np.atleast_2d(weight).astype(np.float32))
    if bias is not None:
        unit.bias.data = torch.tensor(np.atleast_1d(bias).astype(np.float32))
    return unit


def extract_cbp(X, y):
    cbps = []

    # Get the distances
    D = pairwise_distances(X)
    D2 = D * D

    # Extract the class indexes
    klass1 = np.nonzero(y == 1)[0]
    klassm1 = np.nonzero(y == -1)[0]

    # For each point from class 1
    for i in klass1:
        # For each point from class -1
        for j in klassm1:
            # For all the points
            good = True
            for k in range(D.shape[0]):
                if k != i and k != j:
                    # Compute the distance da from i to k and db from j to k
                    da = D2[i, k] * (1 - (D2[j, k] - D2[i, k] - D2[i, j]) / (-2 * D[i, k] * D[i, j]))
                    db = np.square(
                        D[i, k] * ((D2[j, k] - D2[i, k] - D2[i, j]) / (-2 * D[i, k] * D[i, j])) - (D[i, j] / 2))

                    # Compute the distance from the new point k to the center point
                    d_km = np.sqrt(da + db)
                    print(d_km, D[i, j] / 2)
                    if d_km < D[i, j] / 2:
                        good = False
                        break

            # If have not found any other point which is closest to the central point
            if good:
                # Compute the medium point
                x_m = (X[i, :] + X[j, :]) / 2
                cbps.append(x_m)

    if len(cbps) == 0:
        return np.array((0, X.shape[1]))
    else:
        return np.array(cbps)


def compute_maxmin_preact_random(x, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    dim = tuple(set(range(x.ndim)) - {1-ax})
    with torch.no_grad():
        max_ = x.amax(dim=dim, keepdim=True)
        argmaxes = (x == max_).nonzero()
        chosen = []
        for i in range(max_.shape[1-ax]):
            item_maxes = argmaxes[argmaxes[:, 1-ax] == i, ax]
            idx = torch.randint(item_maxes.shape[0], (1,))
            chosen.append(item_maxes[idx])

    chosen = torch.tensor([chosen]) if ax == 0 else torch.tensor([chosen]).T
    max_preact = torch.gather(x, ax, chosen).flatten()

    with torch.no_grad():
        min_ = x.amin(dim=dim, keepdim=True)
        argmines = (x == min_).nonzero()
        chosen = []
        for i in range(min_.shape[1-ax]):
            col_mines = argmines[argmines[:, 1-ax] == i, ax]
            idx = torch.randint(col_mines.shape[0], (1,))
            chosen.append(col_mines[idx])

    chosen = torch.tensor([chosen]) if ax == 0 else torch.tensor([chosen]).T
    min_preact = torch.gather(x, ax, chosen).flatten()

    return max_preact, min_preact


def compute_maxmin_preact_average(x, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    dim = tuple(set(range(x.ndim)) - {1-ax})
    max_preact = x.amax(dim=dim)
    min_preact = x.amin(dim=dim)

    return max_preact, min_preact


def compute_maxmin_preact_single(x, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    x_max = x.max(dim=ax)[0]
    x_min = x.min(dim=ax)[0]

    # Since pytorch can't compute max across multiple dimensions, flatten the rest and aggregate
    max_preact = x_max.view(x_max.shape[0], -1).max(dim=1)[0]
    min_preact = x_min.view(x_min.shape[0], -1).min(dim=1)[0]

    return max_preact, min_preact

def weights_to_df(model):
    layers = {}
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight'):
            weight = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy()
            params = np.concatenate((weight, bias[:, np.newaxis]), axis=1)
            labels = [f'{i}' for i in range(weight.shape[1])]
            labels.append(f'b')
            layers[name] = pd.DataFrame(params, columns=labels).T
            # layers.append(pd.DataFrame(params, columns=labels))

    table = pd.concat(layers).T
    return table
