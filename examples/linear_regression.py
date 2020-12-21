import pandas as pd
from sklearn.model_selection import train_test_split

from cu_ml import LinearRegressionGD

if __name__ == "__main__":
    df = pd.read_csv("sample_data/salary.csv")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1 / 3, random_state=0
    )

    model = LinearRegressionGD(
        num_iterations=100, learning_rate=0.1, backend="numpy", verbose="INFO"
    )

    model.fit(X_train, Y_train)

    model.plot_loss()

    Y_pred = model.predict(X_test)

    print(f"Y_Pred: {Y_pred[:5]}")
    print(f"Y_True: {Y_test[:5]}")
