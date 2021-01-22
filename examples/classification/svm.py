import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from swi_ml.svm import SVM
from swi_ml import set_backend

set_backend("numpy")

if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    model = SVM(num_iterations=1000, learning_rate=0.0001, regularisation_ratio=0.9)
    model.fit(X, y)

    print(model.W, model.b)

    pred = model.predict(X)

    def plot_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X[:,0], X[:,1], marker='o',c=y)

        x0_1 = np.amin(X[:,0])
        x0_2 = np.amax(X[:,0])

        x1_1 = get_hyperplane_value(x0_1, model.W, model.b, 0)
        x1_2 = get_hyperplane_value(x0_2, model.W, model.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, model.W, model.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, model.W, model.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, model.W, model.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, model.W, model.b, 1)

        ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
        ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
        ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

        x1_min = np.amin(X[:,1])
        x1_max = np.amax(X[:,1])
        ax.set_ylim([x1_min-3,x1_max+3])

        plt.show()

    plot_svm()