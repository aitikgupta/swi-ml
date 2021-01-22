import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from swi_ml.classification import LogisticRegressionGD
from swi_ml import set_backend

set_backend("numpy")

if __name__ == "__main__":
    df = pd.read_csv("sample_data/diabetes.csv")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1 / 3, random_state=0
    )

    model = LogisticRegressionGD(
        num_iterations=100, learning_rate=0.01, verbose="INFO"
    )

    print("Input data shape:", X_train.shape, Y_train.shape)
    model.fit(X_train, Y_train)

    model.plot_loss()
    plt.show()

    Y_pred = model.predict(X_test)

    print(f"Y_Pred: {np.round(Y_pred[:5], 0)}")
    print(f"Y_True: {Y_test[:5]}")
