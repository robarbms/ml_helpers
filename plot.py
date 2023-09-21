import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(train_data, train_labels, test_data, test_labels, preditions):
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")

    # Plot testing data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")

    # Plot model's predictions in red
    plt.scatter(test_data, preditions, c="r", label="Predictions")

    # Show the legend
    plt.legend()

def plot_history(history):
    pd.DataFrame(history.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")

