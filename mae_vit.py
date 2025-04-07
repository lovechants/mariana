import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Config
DATA_DIR = Path("your_data_folder")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BANDS = (
    [f"B{i}" for i in range(1, 13)]
    + ["B13lo", "B13hi", "B14lo", "B14hi"]
    + [f"B{i}" for i in range(15, 37)]
)
NUM_BANDS = len(BANDS)

PATCH_SIZE = (4, 4, 4)
MASK_RATIO = 0.75
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Dataset
class HyperspectralDataset(Dataset):
    def __init__(self, folder, supervised=False):
        self.folder = Path(folder)
        self.supervised = supervised
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
        x = torch.tensor(self.inputs[idx])  # (1, C, 5, 5)
        if self.supervised:
            y = torch.tensor(self.labels[idx])
            return x, y
        else:
            return x

# Patch embedding
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, C', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

# MAE-ViT
class MAEViT(nn.Module):
    def __init__(
        self,
        patch_size=(4, 4, 4),
        embed_dim=512,
        encoder_depth=12,
        decoder_dim=256,
        decoder_depth=4,
        num_heads=8,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed3D(patch_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=encoder_depth,
        )
        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decoder_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=decoder_depth,
        )
        patch_vol = np.prod(patch_size)
        self.head = nn.Linear(decoder_dim, patch_vol)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, _ = x.shape

        num_mask = int(self.mask_ratio * N)
        perm = torch.rand(B, N, device=x.device).argsort(dim=1)
        mask_idx = perm[:, :num_mask]
        keep_idx = perm[:, num_mask:]

        x_keep = torch.gather(
            x,
            dim=1,
            index=keep_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )
        enc_out = self.encoder(x_keep)
        dec_tokens = self.enc_to_dec(enc_out)

        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        dec_input = torch.zeros(
            B, N, dec_tokens.size(-1), device=x.device
        )
        dec_input.scatter_(
            1,
            keep_idx.unsqueeze(-1).expand(-1, -1, dec_tokens.size(-1)),
            dec_tokens,
        )
        dec_input.scatter_(
            1,
            mask_idx.unsqueeze(-1).expand(-1, -1, dec_tokens.size(-1)),
            mask_tokens,
        )
        dec_out = self.decoder(dec_input)
        pred = self.head(dec_out)
        return pred, mask_idx

# Pretraining
def pretrain_mae(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb in tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}"):
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            pred, mask_idx = model(xb)
            with torch.no_grad():
                patches_gt = model.patch_embed(xb)
            patch_vol = pred.size(-1)
            target = patches_gt.reshape(patches_gt.size(0), -1, patch_vol)
            mask = torch.zeros_like(pred[..., 0]).bool()
            mask.scatter_(1, mask_idx, True)
            loss = F.mse_loss(pred[mask], target[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Pretrain Epoch {epoch+1}: Loss = {avg_loss:.6f}")

# Finetuning head
class RegressionHead(nn.Module):
    def __init__(self, encoder_dim, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

# Finetuning
def finetune(model, head, train_loader, val_loader, epochs):
    params = list(head.parameters()) + list(model.encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    best_val_loss = float("inf")
    patience = 5
    counter = 0

    for epoch in range(epochs):
        model.eval()
        head.train()
        total_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                tokens = model.encoder(model.patch_embed(xb))
            preds = head(tokens)
            loss = F.mse_loss(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        head.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                tokens = model.encoder(model.patch_embed(xb))
                preds = head(tokens)
                loss = F.mse_loss(preds, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.6f}, Val Loss={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "head": head.state_dict(),
                },
                CHECKPOINT_DIR / "best_mae_vit_regression.pth",
            )
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

if __name__ == "__main__":
    # Pretraining
    pretrain_dataset = HyperspectralDataset(DATA_DIR, supervised=False)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)

    mae_vit = MAEViT(
        patch_size=PATCH_SIZE,
        embed_dim=512,
        encoder_depth=12,
        decoder_dim=256,
        decoder_depth=4,
        num_heads=8,
        mask_ratio=MASK_RATIO,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(mae_vit.parameters(), lr=LR, weight_decay=1e-4)
    pretrain_mae(mae_vit, pretrain_loader, optimizer, PRETRAIN_EPOCHS)

    torch.save(mae_vit.state_dict(), CHECKPOINT_DIR / "mae_vit_pretrained.pth")

    # Finetuning
    folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])
    datasets = [HyperspectralDataset(f, supervised=True) for f in folders]
    full_dataset = ConcatDataset(datasets)
    total = len(full_dataset)
    val_size = int(0.1 * total)
    test_size = int(0.1 * total)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    head = RegressionHead(encoder_dim=512).to(DEVICE)
    finetune(mae_vit, head, train_loader, val_loader, FINETUNE_EPOCHS)

    # Load best model
    checkpoint = torch.load(CHECKPOINT_DIR / "best_mae_vit_regression.pth")
    mae_vit.encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])

    # Test evaluation
    mae_vit.eval()
    head.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            tokens = mae_vit.encoder(mae_vit.patch_embed(xb))
            preds = head(tokens)
            preds_all.append(preds.cpu())
            labels_all.append(yb.cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    mse = np.mean((preds_all - labels_all) ** 2)
    print(f"Test MSE: {mse:.6f}")
