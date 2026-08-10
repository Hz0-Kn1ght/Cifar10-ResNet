import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.figure(figsize=(8,5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(
        "results/loss.png"
    )
    plt.close()

def plot_accuracy(accs):

    plt.figure(figsize=(8,5))

    plt.plot(accs)

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Test Accuracy")

    plt.savefig(
        "results/accuracy.png",
        dpi=300
    )

    plt.close()