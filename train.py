import torch
from torch import nn
from evaluate import evaluate
from utils.visualize import plot_accuracy, plot_loss

def train(model, train_loader, test_loader, device, epochs=10):
    
    model = model.to(device)
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
    )

    train_losses = []
    test_accs = []
    best_acc = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        epoch_loss = total_loss/len(train_loader)
        train_losses.append(epoch_loss)

        acc = evaluate(model, test_loader, device)
        test_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                "checkpoints/resnet18.pth"
            )

        print(
            f"epoch {epoch+1}, "
            f"loss={epoch_loss:.4f}, "
            f"acc={acc:.4f}"
        )
            
    plot_loss(train_losses)
    plot_accuracy(test_accs)