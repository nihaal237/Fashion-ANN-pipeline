from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    preprocess_params = params["preprocess"]

    x_train = np.load(RAW_DIR / "x_train.npy")
    y_train = np.load(RAW_DIR / "y_train.npy")
    x_test = np.load(RAW_DIR / "x_test.npy")
    y_test = np.load(RAW_DIR / "y_test.npy")

    # Main branch normalization: standard [0, 1] scaling with clipping
    x_train = np.clip(x_train.astype("float32") / 255.0, 0.0, 1.0)
    x_test = np.clip(x_test.astype("float32") / 255.0, 0.0, 1.0)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=preprocess_params["test_size"],
        random_state=preprocess_params["seed"],
        stratify=y_train
    )

    np.save(PROCESSED_DIR / "x_train.npy", x_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "x_val.npy", x_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "x_test.npy", x_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print("Preprocessing complete.")
    print("x_train:", x_train.shape)
    print("x_val:", x_val.shape)
    print("x_test:", x_test.shape)
    print("Pixel range:", x_train.min(), "to", x_train.max())


if __name__ == "__main__":
    main()