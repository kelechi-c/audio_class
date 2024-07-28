import torch
import os
from torch import nn, optim
import wandb
from utils import config
from tqdm.auto import tqdm
from audio_classifier import MusiClass
from dataloader import train_loader, valid_loader

classifier = MusiClass()
model = classifier.to(config.device)

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.Adam(params=classifier.parameters(), lr=config.lr)
print(len(classifier.parameters()))

epochs = config.num_epochs

# initilaize wandb
wandb.login()
train_run = wandb.init(project="musiclass", name="musiclass_1")
wandb.watch(classifier, log_freq=100)


os.mkdir(config.model_outpath)
output_path = os.path.join(os.getcwd(), config.model_output_path)


def train_step(train_loader, model, device=config.device):
    total_correct = 0
    total_samples = 0

    for _, (audio, label) in tqdm(enumerate(train_loader)):
        audio = audio.to(device)
        label = label.to(device)

        model_outputs = model(audio)

        _, predicted = torch.max(model_outputs, 1)

        total_correct += (predicted == label).sum().item()
        total_samples += label.size(0)

        print(f"total samples {total_samples}")

        accuracy = 100 * total_correct / total_samples
        train_loss = criterion(model_outputs, label)

        optimizer.zero_grad()
        train_loss.backwards()
        optimizer.step()

    return accuracy, train_loss


def validation_step(model, valid_loader, device=config.device):
    val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for _, (audio, label) in tqdm(enumerate(train_loader)):
            audio = audio.to(device)
            label = label.to(device)

            model_outputs = model(audio)

            _, predicted = torch.max(model_outputs, 1)

            val_loss = criterion(model_outputs, label)

            optimizer.zero_grad()
            val_loss.backwards()
            optimizer.step()

    return val_loss


def training_loop(model, train_loader, valid_loader, epochs=epochs):
    model.train()
    for epoch in tqdm(range(epochs)):
        print(f"Training epoch {epoch}")
        train_acc, train_loss = train_step(train_loader, model)
        valid_loss = validation_step(model, valid_loader)

        print(
            f"Epoch {epoch} of {epochs}, train_accuracy: {train_acc:.4f}, train_loss: {train_loss.item():.4f},  val_loss: {train_loss.item():.2f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(
            checkpoint, os.path.join(output_path, f"musiclass_model_{epoch}.pth")
        )
        print(f"Saved model checkpoint @ epoch {epoch}")

        wandb.log({"accuracy": train_acc, "loss": train_loss, "val_loss": valid_loss})
        print(f"Epoch @ {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, accuracy: {train_acc:.2f}, train_loss: {train_loss.item():.4f},valid_accuracy: {valid_acc:.2f}, valid_loss: {valid_loss:.4f}"
    )

    torch.save(
        model.state_dict(), os.path.join(output_path, f"{config.model_filename}")
    )


training_loop(classifier, train_loader, valid_loader)
