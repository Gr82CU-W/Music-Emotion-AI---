import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random

# =========================
# 1. 特征提取与增强 (优化)
# =========================
def extract_features(y, sr=22050, n_mels=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)
    features = np.stack([mel, delta, delta2], axis=0)

    for i in range(3):
        mean = features[i].mean()
        std = features[i].std()
        if std > 1e-6:
            features[i] = (features[i] - mean) / std
        else:
            features[i] = features[i] - mean

    return features.astype(np.float32)

def spec_augment(spec, freq_mask_param=15, time_mask_param=35):
    """
    SpecAugment: 随机频率和时间遮挡，防止过拟合
    spec: (C, F, T)
    """
    aug_spec = spec.copy()
    C, F, T = aug_spec.shape
    
    # 频率遮挡 (Frequency Masking)
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, F - f)
    aug_spec[:, f0:f0+f, :] = 0
    
    # 时间遮挡 (Time Masking)
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, T - t)
    aug_spec[:, :, t0:t0+t] = 0
    
    return aug_spec

# =========================
# 2. 模型（引入 SE-Block 注意力机制）
# =========================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: 通道注意力机制"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True), # 使用 SiLU 替代 ReLU
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) # 升级激活函数
        
        # 引入 SE 注意力
        self.se = SEBlock(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # 应用注意力
        out += self.shortcut(x)
        return self.act(out)

class SimpleCNNEmotion(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5) # 增加 Dropout 防止过拟合
        )

        # gate：象限 logits（软监督）
        self.gate = nn.Linear(128, 4)

        # 4 experts：直接回归 logit VA（不 tanh）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 2)
            ) for _ in range(4)
        ])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        h = self.embed(x)
        gate_logits = self.gate(h)
        gate_p = torch.softmax(gate_logits, dim=1)

        expert_out = torch.stack([e(h) for e in self.experts], dim=1)
        reg_logit = (gate_p.unsqueeze(-1) * expert_out).sum(dim=1)
        reg_out = torch.sigmoid(reg_logit)

        return reg_out, gate_logits

# =========================
# 3. Dataset（不变）
# =========================
class MusicDataset(Dataset):
    def __init__(self, paths, labels, sr=22050, duration=10, train=True):
        self.paths = paths
        self.labels = labels
        self.sr = sr
        self.target_len = int(sr * duration)
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            y, _ = librosa.load(self.paths[idx], sr=self.sr)
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            y = np.zeros(self.target_len)

        if len(y) > self.target_len:
            # 训练时随机裁剪，验证时中心裁剪
            if self.train:
                start = np.random.randint(0, len(y) - self.target_len)
                y = y[start : start + self.target_len]
            else:
                c = len(y) // 2
                y = y[c - self.target_len // 2: c + self.target_len // 2]
        else:
            y = librosa.util.fix_length(y, size=self.target_len)

        feat = extract_features(y)
        feat = librosa.util.fix_length(feat, size=430, axis=2)
        
        # 训练时应用 SpecAugment
        if self.train:
            feat = spec_augment(feat)
            
        return torch.tensor(feat), torch.tensor(self.labels[idx], dtype=torch.float32)

# =========================
# 4. 软象限标签（新）
# =========================
def soft_quadrant_labels(targets):
    v, a = targets[:, 0], targets[:, 1]
    q = torch.stack([
        v * a,
        (1 - v) * a,
        (1 - v) * (1 - a),
        v * (1 - a)
    ], dim=1)
    return q / (q.sum(dim=1, keepdim=True) + 1e-8)

def decorrelation_loss(v, a):
    v0 = v - v.mean()
    a0 = a - a.mean()
    return torch.abs((v0 * a0).mean())

# =========================
# 4.5 数据加载 (新)
# =========================
def get_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "audiolist")
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found.")
        return [], [], [], []

    all_files = []
    
    # 遍历 audiolist 文件夹
    print(f"正在扫描目录: {root_dir}")
    total_scanned = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                total_scanned += 1
                # 解析文件名: 音乐名_V值_A值_实际分类.mp3
                try:
                    name_parts = os.path.splitext(file)[0].split('_')
                    if len(name_parts) >= 4:
                        # 倒数第3个是V，倒数第2个是A
                        v_str = name_parts[-3]
                        a_str = name_parts[-2]
                        
                        # 处理可能存在的 'V' 或 'A' 前缀
                        if v_str.upper().startswith('V'): v_str = v_str[1:]
                        if a_str.upper().startswith('A'): a_str = a_str[1:]
                        
                        v_val = float(v_str)
                        a_val = float(a_str)
                        
                        # 归一化 0-10 -> 0-1
                        v_norm = v_val / 10.0
                        a_norm = a_val / 10.0
                        
                        v_norm = max(0.0, min(1.0, v_norm))
                        a_norm = max(0.0, min(1.0, a_norm))
                        
                        # 确定象限 (以5.0为界)
                        if v_val >= 5.0 and a_val >= 5.0: q = 0
                        elif v_val < 5.0 and a_val >= 5.0: q = 1
                        elif v_val < 5.0 and a_val < 5.0: q = 2
                        else: q = 3
                        
                        # 计算距离中心(5,5)的距离，用于筛选最典型的样本
                        dist = np.sqrt((v_val - 5.0)**2 + (a_val - 5.0)**2)
                        
                        all_files.append({
                            'path': os.path.join(root, file),
                            'v': v_norm,
                            'a': a_norm,
                            'q': q,
                            'dist': dist,
                            'filename': file
                        })
                except Exception as e:
                    # print(f"解析失败 {file}: {e}")
                    pass
    
    print(f"扫描完成。共发现 {total_scanned} 个音频文件，其中 {len(all_files)} 个符合命名规范。")

    # 保存记录到 CSV
    if all_files:
        df = pd.DataFrame(all_files)
        df.to_csv('dataset_processed.csv', index=False)
        print(f"共找到 {len(df)} 个有效音频文件，已保存至 dataset_processed.csv")
    else:
        print("未找到符合命名规范的音频文件。")
        return [], [], [], []

    # 数据筛选逻辑
    train_data = []
    val_data = []
    
    target_train = 300
    target_val = 25
    
    print("开始划分数据集 (每类目标: 训练300, 验证25)...")
    
    for q in range(4):
        items = [x for x in all_files if x['q'] == q]
        print(f"象限 {q} (原始数量): {len(items)}")
        
        # 按距离降序排列 (优先选择远离中心的典型样本)
        items.sort(key=lambda x: x['dist'], reverse=True)
        
        # 选取样本
        total_needed = target_train + target_val
        
        if len(items) > total_needed:
            selected = items[:total_needed]
        else:
            selected = items
            
        # 划分训练和验证
        if len(selected) >= target_train:
            curr_train = selected[:target_train]
            curr_val = selected[target_train:]
            if len(curr_val) > target_val:
                curr_val = curr_val[:target_val]
        else:
            curr_train = selected
            curr_val = []
        
        train_data.extend(curr_train)
        val_data.extend(curr_val)
        
        print(f"  -> 最终选取: 训练 {len(curr_train)}, 验证 {len(curr_val)}")

    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return (
        [x['path'] for x in train_data],
        [[x['v'], x['a']] for x in train_data],
        [x['path'] for x in val_data],
        [[x['v'], x['a']] for x in val_data]
    )

# =========================
# 5. 训练函数（仅 loss 改动）
# =========================
def train_model(train_loader, val_loader, device, resume=False):
    model = SimpleCNNEmotion().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    reg_loss = nn.SmoothL1Loss()
    kl = nn.KLDivLoss(reduction='batchmean')
    
    save_path = "model_music2emo_unified.pth"
    checkpoint_path = "checkpoint_latest.pth"
    best_val_loss = float('inf')
    start_epoch = 0

    # Resume 逻辑
    if resume and os.path.exists(checkpoint_path):
        print(f"正在从 {checkpoint_path} 恢复训练...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"已恢复至 Epoch {start_epoch}")
    elif os.path.exists(save_path):
        print(f"加载已有模型权重 {save_path} 作为起点...")
        try:
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        except:
            print("权重加载失败，将从头开始训练。")

    print(f"开始训练... 模型将保存至: {save_path}")

    try:
        for epoch in range(start_epoch, 100):
            model.train()
            train_loss = 0.0
            
            # 训练循环
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/100")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                pred, gate_logits = model(x)
                soft_q = soft_quadrant_labels(y)

                loss_reg = reg_loss(pred, y)
                loss_gate = kl(F.log_softmax(gate_logits, dim=1), soft_q)
                loss_decorr = decorrelation_loss(pred[:, 0], pred[:, 1])

                loss = loss_reg + 0.5 * loss_gate + 0.05 * loss_decorr

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            
            # 验证循环
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred, _ = model(x)
                    loss_reg = reg_loss(pred, y)
                    val_loss += loss_reg.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"✓ 验证集 Loss 降低，模型已保存至 {save_path}")
            
            # 保存 Checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, checkpoint_path)

    except KeyboardInterrupt:
        print("\n\n[警告] 检测到用户中断训练 (Ctrl+C)。")
        print(f"正在紧急保存当前 Checkpoint 至: {checkpoint_path} ...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print("保存完成。下次运行时可选择恢复训练。")
        
    return model

# ===========================
# 6. 预测函数
# ===========================
def predict_emotion(model_path, audio_path, sr=22050, duration=10.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleCNNEmotion()
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except Exception as e:
            print(f"模型加载失败：{e}")
            return None
    else:
        print("未找到模型文件。")
        return None
    
    model.to(device)
    model.eval()

    print(f"正在处理音频: {audio_path} ...")
    try:
        y, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"无法读取文件: {e}")
        return None

    target_len = int(sr * duration)
    
    if len(y) > target_len:
        center = len(y) // 2
        start = center - (target_len // 2)
        y = y[start : start + target_len]
    elif len(y) < target_len:
        y = librosa.util.fix_length(y, size=target_len)
        
    features = extract_features(y, sr=sr, n_mels=128)
    target_frames = 430
    features = librosa.util.fix_length(features, size=target_frames, axis=2)

    # 4. 预测
    # 增加 Batch 维度: (3, 128, 430) -> (1, 3, 128, 430)
    input_tensor = torch.tensor(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reg_out, _ = model(input_tensor)
        result = reg_out.cpu().numpy()[0] # 取出第一个样本的回归结果

    return result

if __name__ == "__main__":
    print("--- 音乐情绪识别程序 (Updated) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = "model_music2emo_unified.pth"
    checkpoint_file = "checkpoint_latest.pth"
    
    import sys
    mode = "train"
    if len(sys.argv) > 1:
        if sys.argv[1] == "predict":
            mode = "predict"
    
    if mode == "train":
        resume = False
        if os.path.exists(checkpoint_file):
            # 自动检测是否需要 resume，或者通过参数控制
            if "--resume" in sys.argv:
                resume = True
            else:
                print(f"提示: 检测到 {checkpoint_file}，如需恢复训练请使用 python music2emo.py --resume")
        
        print("准备开始训练...")
        train_paths, train_labels, val_paths, val_labels = get_data()
        
        if not train_paths:
            print("未找到有效数据，请检查 audiolist 文件夹及文件名格式。")
        else:
            print(f"训练集: {len(train_paths)}, 验证集: {len(val_paths)}")
            # 训练集开启数据增强 (train=True)
            train_dataset = MusicDataset(train_paths, train_labels, train=True)
            # 验证集关闭数据增强 (train=False)
            val_dataset = MusicDataset(val_paths, val_labels, train=False)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
            
            model = train_model(train_loader, val_loader, device, resume=resume)
            
            torch.save(model.state_dict(), model_file)
            print(f"训练完成，模型已保存至: {model_file}")

    elif mode == "predict":
        test_file = "test_music.mp3"
        if len(sys.argv) > 2:
            test_file = sys.argv[2]
            
        if os.path.exists(test_file):
            result = predict_emotion(model_file, test_file)
            if result is not None:
                v, a = result
                v_scaled = v * 10.0
                a_scaled = a * 10.0
                print(f"预测结果 (0-10): Valence={v_scaled:.2f}, Arousal={a_scaled:.2f}")
        else:
            print(f"文件不存在: {test_file}")