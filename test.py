import torch

def test(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            imgaes = images.to(device)
            labels = labels.to(device)

            outputs = model(imgaes)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}")