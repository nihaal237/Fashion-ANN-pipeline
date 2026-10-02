from pathlib import Path
import numpy as np
from tensorflow import keras


RAW_DIR = Path("data/raw")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    np.save(RAW_DIR / "x_train.npy", x_train)
    np.save(RAW_DIR / "y_train.npy", y_train)
    np.save(RAW_DIR / "x_test.npy", x_test)
    np.save(RAW_DIR / "y_test.npy", y_test)

    print("Fashion-MNIST downloaded and saved to data/raw/")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)


if __name__ == "__main__":
    main()