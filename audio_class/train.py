import torch
import os
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from utils import config, count_params
from tqdm.auto import tqdm
from audio_classifier import MusiClass
from dataloader import train_loader

classifier = MusiClass()
classifier = classifier.to(config.device)

scaler = GradScaler()


criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.Adam(params=classifier.parameters(), lr=config.lr)
param_count = count_params(classifier)

print(f"Model parameters => {param_count}")

epochs = config.epochs

# initilaize wandb
wandb.login()
train_run = wandb.init(project="musiclass", name="musiclass_1")
wandb.watch(classifier, log_freq=100)


os.mkdir(config.model_outpath)
output_path = os.path.join(os.getcwd(), config.model_outpath)


def training_loop(
    model=classifier, train_loader=train_loader, epochs=epochs, config=config
):
    model.train()
    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        print(f"Training epoch {epoch}")

        train_loss = 0.0

        for _, (audio, label) in tqdm(enumerate(train_loader)):
            model_outputs = model(audio)

            train_loss = criterion(model_outputs, label)
            optimizer.zero_grad()

            train_loss.backwards()
            optimizer.step()

            torch.cuda.empty_cache()

        print(f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(
            checkpoint, os.path.join(output_path, f"musiclass_model_{epoch}.pth")
        )

        print(f"Saved model checkpoint @ epoch {epoch}")

        wandb.log({"loss": train_loss})

        print(f"Epoch @ {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, accuracy: {train_acc:.2f}, train_loss: {train_loss.item():.4f},valid_accuracy: {valid_acc:.2f}, valid_loss: {valid_loss:.4f}"
    )

    torch.save(
        model.state_dict(), os.path.join(output_path, f"{config.model_filename}")
    )


training_loop()
print("music classifier training complete")

training_loop(classifier, train_loader)
