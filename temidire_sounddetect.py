import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import os
from torch import nn, optim
from torch.nn import functional as fnn
from torch.cuda.amp import autocast, GradScaler  # , init_scale
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import gc
import wandb


class config:
    batch_size = 1
    lr = 1e-3
    grad_acc_step = 4
    max_duration = 30
    epochs = 50
    num_classes = 18
    split = 100
    sr = 16000
    train_split = 0.9
    target_size = 475000
    audio_length = 1000
    dataset_id = ""
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_outpath = "sound_detect"
    model_filename = "sound_detect.pth"


split = 500

hfdata = load_dataset(config.dataset_id, split="train", streaming=True)
hfdata = hfdata.take(split)


class SoundData(IterableDataset):
    def __init__(self, dataset=hfdata):
        self.dataset = dataset
        self.target_shape = config.target_size

    def __iter__(self):
        for item in self.dataset:
            audio = item["audio"]["array"]

            audio = librosa.resample(
                audio, orig_sr=item["audio"]["sampling_rate"], target_sr=config.small_sr
            )
            audio = librosa.util.fix_length(
                audio, size=self.target_shape, axis=0)

            label = item["genre_id"]

            yield audio, label


m_data = SoundData()

train_loader = DataLoader(m_data, batch_size=config.batch_size)
print("dataloader created")

x = next(iter(train_loader))[0]
print(x)
# model definition


class MusiClass(nn.Module):
    def __init__(self, out_classes=18):
        super().__init__()
        self.audio_convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        with torch.no_grad():
            sample_input = torch.randn(1, 1, 128, 937)
            conv_output = self.audio_convnet(sample_input)
            fs = conv_output.view(1, -1).size(1)

        self.batch_norm = nn.BatchNorm2d(128)

        self.linear_fc = nn.Sequential(nn.Linear(fs, 256), nn.ReLU())

        self.l2 = nn.Linear(256, out_classes)

    def _spectrogram(self, audio, config=config):
        if not isinstance(audio, np.ndarray):
            audio = audio.cpu().numpy()

        n_frames = int(librosa.time_to_frames(
            config.max_duration, sr=config.small_sr))
        audio = audio.squeeze(0)
        specgram = librosa.feature.melspectrogram(
            y=audio, sr=config.sr, n_mels=128, fmax=8000
        )
        specgram = librosa.power_to_db(specgram, ref=np.max)

        if specgram.shape[1] < n_frames:
            specgram = librosa.util.fix_length(specgram, size=n_frames, axis=1)
        else:
            specgram = specgram[:, :n_frames]

        specgram = torch.tensor(specgram, dtype=config.dtype).to(config.device)

        return specgram

    def forward(self, x: torch.Tensor):
        x = self._spectrogram(x)
        x = x.unsqueeze(0)  # Add channel dimension
        x = self.audio_convnet(x)

        x = torch.flatten(x)

        x = self.linear_fc(x)
        x = self.l2(x)
        x = x.unsqueeze(0)
        return x  # Shape: (batch_size, out_classes)


classifier = MusiClass()

classifier = classifier.to(config.dtype).to(config.device)
classifier = torch.compile(classifier)

x = next(iter(train_loader))[0]
print(x.shape)
torch.cuda.empty_cache()

k = classifier(x)

torch.cuda.empty_cache()
gc.collect()

print(f"model pred shape => {k.shape}")
print(k)


criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.Adam(params=classifier.parameters(), lr=config.lr)
scaler = GradScaler()

# param_count = count_params(classifier)
# print(param_count)

epochs = config.epochs

# initilaize wandb
# wandb.login(key=wandb_key)
# train_run = wandb.init(project="musiclass", name="musiclass_1")
# wandb.watch(classifier, log_freq=100)


if os.path.exists(config.model_outpath) is not True:
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

        for x, (audio, label) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = audio.to(config.device)
            label = label.to(config.device)

            # every iterations
            torch.cuda.empty_cache()
            gc.collect()

            # Mixed precision training

            with autocast():
                outputs = model(audio)
                train_loss = criterion(outputs, label.long())
                train_loss = train_loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

        print(f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}")

        #         checkpoint = {
        #             "epoch": epoch,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #         }

        #         torch.save(
        # #             checkpoint, os.path.join(output_path, f"sounddetect_model_{epoch}.pth")
        #         )

        #         print(f"Saved model checkpoint @ epoch {epoch}")

        wandb.log({"loss": train_loss})

        print(f"Epoch @ {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}")

    torch.save(
        model.state_dict(), os.path.join(
            output_path, f"{config.model_filename}")
    )


training_loop()

torch.cuda.empty_cache()
gc.collect()

print("sound class detection training complete")


########################################################################

DEFAULT_CHANNEL_AND_POOL = [(64, 2), (128, 2), (256, 2), (512, 1)]


def interpolate(x, ratio):
    """
    Upscales the 2nd axis of x by 'ratio', i.e Repeats each element in it 'ratio' times:
    In other words: Interpolate the prediction to have the same time_steps as the target.
    The time_steps mismatch is caused by maxpooling in CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to upsample
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def init_layer(layer, nonlinearity="leaky_relu"):
    """Initialize a Linear or Convolutional layer."""
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""

    bn.bias.data.fill_(0.0)
    bn.running_mean.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_var.data.fill_(1.0)


class MobileNetV1(nn.Module):
    def __init__(self, classes_num):
        super(MobileNetV1, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            return _layers

        self.num_pools = 3
        self.features = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1),
            conv_dw(1024, 1024, 1),
        )
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        """
        Input: (batch_size, data_length)"""
        x = x.transpose(
            0, 1)  # -> (batch_size, channels_num, times_steps, freq_bins)
        # x = x.transpose(1, 3)
        # x = self.bn0(x)
        # x = x.transpose(1, 3)

        x = self.features(x)  # (batch_size, 512, time_steps / x, mel_bins / x)
        x = torch.mean(x, dim=3)  # (batch_size, 512, time_steps / x)

        x = x.transpose(1, 2)  # (batch_size, time_steps, 512)
        x = F.relu_(self.fc1(x))  # (batch_size, time_steps, 512)
        # embedding = F.dropout(x, p=0.5, training=self.training)

        # (batch_size, time_steps, classes_num)
        event_output = torch.sigmoid(self.fc_audioset(x))

        # Interpolate
        event_output = interpolate(event_output, 2**self.num_pools)

        return event_output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=2):
        super(ConvBlock, self).__init__()
        self.pool_size = pool_size
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        x = F.avg_pool2d(x, kernel_size=self.pool_size)

        return x


class Cnn_AvgPooling(nn.Module):
    def __init__(self, classes_num, model_config=DEFAULT_CHANNEL_AND_POOL):
        super(Cnn_AvgPooling, self).__init__()
        self.model_config = model_config
        self.num_pools = 1 if model_config[0][1] == 2 else 1
        self.conv_blocks = [
            ConvBlock(
                in_channels=audio_channels,
                out_channels=model_config[0][0],
                pool_size=model_config[0][1],
            )
        ]
        for i in range(1, len(model_config)):
            pool_size = model_config[i][1]
            if pool_size == 2:
                self.num_pools += 1
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=model_config[i - 1][0],
                    out_channels=model_config[i][0],
                    pool_size=pool_size,
                )
            )

        self.conv_blocks = torch.nn.Sequential(*self.conv_blocks)

        self.event_fc = nn.Linear(model_config[-1][0], classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.event_fc)

    def forward(self, x):
        """
        Input: (batch_size, channels_num, times_steps, freq_bins)"""

        x = self.conv_blocks(x)

        # x.shape : (batch_size, channels_num, times_steps, freq_bins)

        x = torch.mean(x, dim=3)  # (batch_size, channels_num, time_steps)
        x = x.transpose(1, 2)  # (batch_size, time_steps, channels_num)

        # event_output = torch.sigmoid(self.event_fc(x))  # (batch_size, time_steps, classes_num)
        # (batch_size, time_steps, classes_num)
        event_output = self.event_fc(x)

        # Interpolate
        event_output = interpolate(event_output, 2 ** (self.num_pools))

        return event_output

    def logits(self, x):
        return torch.sigmoid(self.forward(x))
