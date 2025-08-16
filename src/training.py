def train(dataloader, model, device):
    model.train()
    total_loss = 0

    for batch_idx, (X, y) in enumerate(dataloader):
        size = len(dataloader.dataset)
        X, y = X.to(device), y.to(device)

        prediction = model(X)
        loss = model.loss_fn(prediction, y)

        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(X)
            total_loss += loss
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / len(dataloader)