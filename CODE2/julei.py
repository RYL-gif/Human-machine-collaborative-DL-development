import numpy as np
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
from kmeans import lloyd

def julei(X, num, function):
    #输入dataset,聚类数量，聚类方式
    if function == 'kmeans':
        #使用skleran聚类
        
        km = KMeans(n_clusters=num,init="random")
        km.fit(X)
        choice_cluster = km.predict(X)
        
        #使用外部代码聚类-快
        #choice_cluster, initial_state = lloyd(X, num)
        return choice_cluster

    if function == 'cmeans':
        X = X.T
        center, u, u0, d, jm, p, fpc = cmeans(X, m=2, c=3, error=0.005, maxiter=1000)
        for i in u:
            choice_cluster = np.argmax(u, axis=0)
        return choice_cluster

    if function == 'GMM':
        from sklearn import mixture
        gmm = mixture.GaussianMixture(n_components=num, covariance_type='spherical').fit(X)
        choice_cluster = gmm.predict(X)
        return choice_cluster
