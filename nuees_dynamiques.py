import numpy as np

class NueesDynamiques:
    def __init__(self, K=3, ni=5, max_iter=50, tol=1e-4):
        self.K = K
        self.ni = ni
        self.max_iter = max_iter
        self.tol = tol

    def initialisation(self, X):
        n = X.shape[0]
        indices = np.random.permutation(n)
        self.E = []
        idx = 0
        for _ in range(self.K):
            self.E.append(X[indices[idx:idx+self.ni]])
            idx += self.ni

    def dist_point_cluster(self, x, Ei):
        return np.mean(np.linalg.norm(Ei - x, axis=1))

    def affectation(self, X):
        C = [[] for _ in range(self.K)]
        for x in X:
            distances = [self.dist_point_cluster(x, Ei) for Ei in self.E]
            i = np.argmin(distances)
            C[i].append(x)
        return C

    def mise_a_jour_etalons(self, Ci):
        Ci = np.array(Ci)
        centroid = Ci.mean(axis=0)
        distances = np.linalg.norm(Ci - centroid, axis=1)
        idx = np.argsort(distances)
        return Ci[idx[:self.ni]]

    def fit(self, X):
        self.initialisation(X)
        for iter in range(self.max_iter):
            C = self.affectation(X)
            newE = []
            for i in range(self.K):
                if len(C[i]) < self.ni:
                    newE.append(self.E[i])
                else:
                    newE.append(self.mise_a_jour_etalons(C[i]))

            delta = sum(np.linalg.norm(self.E[i] - newE[i]) for i in range(self.K))
            self.E = newE

            if delta < self.tol:
                print(f"Convergence atteinte à l'itération {iter}")
                break

        self.C = C
        return C

