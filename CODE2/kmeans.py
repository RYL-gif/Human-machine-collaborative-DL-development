import torch
import numpy as np
from pairwise import pairwise_distance


def forgy(X, n_clusters):
    _len = len(X)
    #随机寻找中心点
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def lloyd(X, n_clusters, device=0, tol=1e-4):
    # 3000 *2
    #把X转为Tensor
    X = torch.from_numpy(X).float()
    #X=torch.tensor(X, dtype=torch.float)
    # 4*2 中心点位置
    initial_state = forgy(X, n_clusters)

    while True:
        # 3000*4 计算距离
        dis = pairwise_distance(X, initial_state)
        # 3000 选择的分类
        choice_cluster = torch.argmin(dis, dim=1)
        # 4*2
        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            # 1950
            selected = torch.nonzero(choice_cluster == index).squeeze()
            # 1950*2
            selected = torch.index_select(X, 0, selected)
            # 4*2
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu().numpy(), initial_state.cpu().numpy()
