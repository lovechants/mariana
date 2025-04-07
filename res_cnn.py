import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BANDS = (
    [f"B{i}" for i in range(1, 13)]
    + ["B13lo", "B13hi", "B14lo", "B14hi"]
    + [f"B{i}" for i in range(15, 37)]
)
NUM_BANDS = len(BANDS)

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Dataset
# =========================
class CubeRegressionDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.inputs, self.labels, self.coords = self.load_data()

    def load_data(self):
        band_arrays = [
            np.load(self.folder / f"{band}.npy").astype(np.float32)
            for band in BANDS
        ]
        cube = np.stack(band_arrays, axis=0)
        label = np.load(self.folder / "x.npy").astype(np.float32)
        mask = ~np.isnan(label)
        indices = np.argwhere(mask)

        samples, targets, coords = [], [], []
        H, W = label.shape
        for y, x in indices:
            if y - 2 < 0 or y + 3 > H or x - 2 < 0 or x + 3 > W:
                continue
            patch = cube[:, y - 2 : y + 3, x - 2 : x + 3]
            if patch.shape != (NUM_BANDS, 5, 5):
                continue
            if np.isnan(patch).any():
                continue
            samples.append(np.expand_dims(patch, 0))
            targets.append(label[y, x])
            coords.append((y, x))
        return (
            np.stack(samples),
            np.array(targets, dtype=np.float32),
            coords,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])

# =========================
# Model
# =========================
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 3, stride, 1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, 3, 1, 1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, 1, stride, bias=False
                ),
                nn.GroupNorm(groups, out_channels),
            )

    def forward(self, x):
        out = F.leaky_relu(self.norm1(self.conv1(x)), 0.1)
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out

class DeepResidual3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_bands=36):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock3D(64, 128, stride=2),
            ResidualBlock3D(128, 128),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock3D(128, 256, stride=2),
            ResidualBlock3D(256, 256),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock3D(256, 512, stride=2),
            ResidualBlock3D(512, 512),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock3D(512, 1024, stride=2),
            ResidualBlock3D(1024, 1024),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out)
        out = self.head(out)
        return out.squeeze(-1)

# =========================
# Loss
# =========================
class WeightedMSELoss(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, preds, targets):
        weights = targets ** self.power
        return torch.mean(weights * (preds - targets) ** 2)

# =========================
# Data preparation
# =========================
def prepare_datasets():
    folders = sorted([f for f in PROCESSED_DIR.iterdir() if f.is_dir()])
    datasets = [CubeRegressionDataset(f) for f in folders]

    full_dataset = ConcatDataset(datasets)
    total = len(full_dataset)
    val_size = int(0.1 * total)
    test_size = int(0.1 * total)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    return train_set, val_set, test_set

# =========================
# Training and validation
# =========================
def train_and_validate():
    train_set, val_set, test_set = prepare_datasets()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = DeepResidual3DCNN(in_channels=1, num_bands=NUM_BANDS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = WeightedMSELoss(power=2)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model for testing
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth"))
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Testing"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            preds_all.append(preds.cpu())
            labels_all.append(yb.cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    mse = np.mean((preds_all - labels_all) ** 2)
    print(f"Test MSE: {mse:.6f}")

    plt.scatter(labels_all, preds_all, alpha=0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Test Set Predictions vs Ground Truth")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_and_validate()
