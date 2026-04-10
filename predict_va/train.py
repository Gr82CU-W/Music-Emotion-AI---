# train.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets.emobank_dataset import EmoBankDataset
from models.va_regressor import VARegressor
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 1. 加载数据
df = pd.read_csv("data/emobank.csv", index_col=0)
train_df = df[df["split"] == "train"]
dev_df = df[df["split"] == "dev"]

# 2. tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./pretrained/bert-base-uncased",
    local_files_only=True
)

# 3. Dataset & DataLoader
train_ds = EmoBankDataset(train_df, tokenizer)
dev_ds = EmoBankDataset(dev_df, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=32)

# 4. 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VARegressor("./pretrained/bert-base-uncased").to(device)

# 5. 优化器 & loss
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.MSELoss()

# 创建结果目录
RESULTS_DIR = "./results/en_training"
os.makedirs(RESULTS_DIR, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/en_train")

# 记录训练过程
train_losses = []
dev_mse_v = []
dev_mse_a = []
dev_r_v = []
dev_r_a = []

def pearson_r(x, y):
    if len(x) < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / denom) if denom != 0 else float("nan")

# 6. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        v = batch["valence"].to(device)
        a = batch["arousal"].to(device)

        v_pred, a_pred = model(ids, mask)
        loss_v = criterion(v_pred, v)
        loss_a = criterion(a_pred, a)
        loss = loss_v + loss_a

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        writer.add_scalar("train/batch_loss", loss.item(), epoch * len(train_loader) + len(batch))

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    writer.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)

    # 验证集评估
    model.eval()
    v_preds_all = []
    a_preds_all = []
    v_trues_all = []
    a_trues_all = []

    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            v = batch["valence"].to(device)
            a = batch["arousal"].to(device)

            v_pred, a_pred = model(ids, mask)
            v_preds_all.extend(v_pred.cpu().numpy().ravel().tolist())
            a_preds_all.extend(a_pred.cpu().numpy().ravel().tolist())
            v_trues_all.extend(v.cpu().numpy().ravel().tolist())
            a_trues_all.extend(a.cpu().numpy().ravel().tolist())

    v_preds_np = np.array(v_preds_all)
    a_preds_np = np.array(a_preds_all)
    v_trues_np = np.array(v_trues_all)
    a_trues_np = np.array(a_trues_all)

    mse_v = float(np.mean((v_preds_np - v_trues_np) ** 2))
    mse_a = float(np.mean((a_preds_np - a_trues_np) ** 2))
    r_v = pearson_r(v_preds_np, v_trues_np)
    r_a = pearson_r(a_preds_np, a_trues_np)

    dev_mse_v.append(mse_v)
    dev_mse_a.append(mse_a)
    dev_r_v.append(r_v)
    dev_r_a.append(r_a)

    writer.add_scalar("val/mse_valence", mse_v, epoch + 1)
    writer.add_scalar("val/mse_arousal", mse_a, epoch + 1)
    writer.add_scalar("val/pearson_valence", r_v, epoch + 1)
    writer.add_scalar("val/pearson_arousal", r_a, epoch + 1)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val MSE (v/a): {mse_v:.4f}/{mse_a:.4f} - Val R (v/a): {r_v:.4f}/{r_a:.4f}")

# 保存训练和验证曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss per Epoch")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "train_loss_curve.png"))
plt.close()

plt.figure()
plt.plot(range(1, num_epochs + 1), dev_mse_v, label="MSE Valence", marker='o')
plt.plot(range(1, num_epochs + 1), dev_mse_a, label="MSE Arousal", marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Validation MSE per Epoch")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "val_mse_curve.png"))
plt.close()

plt.figure()
plt.plot(range(1, num_epochs + 1), dev_r_v, label="Pearson Valence", marker='o')
plt.plot(range(1, num_epochs + 1), dev_r_a, label="Pearson Arousal", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Pearson r")
plt.title("Validation Pearson r per Epoch")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "val_pearson_curve.png"))
plt.close()

# 保存模型
torch.save(model.state_dict(), "va_model.pt")
print("Model saved to va_model.pt")

writer.close()
