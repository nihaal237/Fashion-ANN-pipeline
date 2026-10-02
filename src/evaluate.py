from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def main():
    x_test = np.load(PROCESSED_DIR / "x_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    model = keras.models.load_model(MODEL_DIR / "model.h5")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png")
    plt.close()

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete.")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Metrics saved to metrics.json")
    print("Confusion matrix saved to models/confusion_matrix.png")


if __name__ == "__main__":
    main()