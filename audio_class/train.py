import torch
import gc
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
scaler = GradScaler()

param_count = count_params(classifier)
print(param_count)

epochs = config.epochs

# initilaize wandb
wandb.login()
train_run = wandb.init(project="musiclass", name="musiclass_1")
wandb.watch(classifier, log_freq=100)


if os.path.exists(config.model_outpath) is not True:
    os.mkdir(config.model_outpath)

output_path = os.path.join(os.getcwd(), config.model_outpath)


def training_loop(model=classifier, train_loader=train_loader, epochs=epochs, config=config):
    model.train()
    train_loss = 0.0

    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        print(f"Training epoch {epoch}")

        for x, (audio, label) in tqdm(enumerate(train_loader), total=config.batch_size:
            optimizer.zero_grad()

            audio=audio.to(config.device)
            label=label.to(config.device)

            # every iterations
            torch.cuda.empty_cache()
            gc.collect()

            # Mixed precision training

            with autocast():
                outputs=model(audio)
                train_loss=criterion(outputs, label.long())
                train_loss=train_loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

        if epoch % 5 == 0:

            checkpoint={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(
                checkpoint, os.path.join(
                    output_path, f"musiclass_model_{epoch}.pth")
            )

            print(f"Saved model checkpoint @ epoch {epoch}")

        print(
            f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}"
        )

        wandb.log({"loss": train_loss})

        print(f"Epoch @ {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}"
    )

    torch.save(
        model.state_dict(), os.path.join(
            output_path, f"{config.model_filename}")
    )


training_loop()

torch.cuda.empty_cache()
gc.collect()

print('music classifier training complete')
