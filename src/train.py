from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    x_train = np.load(PROCESSED_DIR / "x_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    x_val = np.load(PROCESSED_DIR / "x_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=64
    )

    model.save(MODEL_DIR / "model.h5")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODEL_DIR / "history.csv", index=False)

    print("Training complete.")
    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")


if __name__ == "__main__":
    main()