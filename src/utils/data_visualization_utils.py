import os

import matplotlib.pyplot as plt

NUMBER_OF_DATA_POINTS = 1_000

PLOT_FLAG = False


def plot_results(loss, val_loss, accuracy, val_accuracy, utc_timestamp):
    results_dir = './results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.figure()
    plt.plot(loss, "r", label="Training loss")
    plt.plot(val_loss, "bo", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    fn = "./results/results_train_val_loss_{}_{}.jpg".format(
        NUMBER_OF_DATA_POINTS, utc_timestamp
    )
    plt.savefig(fn)
    print("results `train_val_loss` saved at: {}".format(fn))
    if PLOT_FLAG:
        plt.show()

    plt.figure()
    plt.plot(accuracy, "r", label="Training accuracy")
    plt.plot(val_accuracy, "bo", label="Validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    fn = "./results/results_train_val_acc_{}_{}.jpg".format(
        NUMBER_OF_DATA_POINTS, utc_timestamp
    )
    plt.savefig(fn)
    print("results `train_val_acc` saved at: {}".format(fn))
    if PLOT_FLAG:
        plt.show()
