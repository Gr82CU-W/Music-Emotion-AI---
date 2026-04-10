import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 检查科学计算库依赖
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr
except ImportError:
    print("错误：缺少必要的科学计算库。")
    print("请在终端运行以下命令安装：")
    print("pip install scikit-learn matplotlib scipy")
    exit()

import sys

# 导入项目核心模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from core.model_audio_cnn import SimpleCNNEmotion, MusicDataset, get_data
except ImportError:
    print(f"错误：无法从 core 导入模块。请确保在项目根目录下运行。")
    exit()
except ImportError:
    print("错误：无法导入 music2emo.py。请确保该文件在当前目录下。")
    exit()

def evaluate_scientific(model_path, batch_size=64):
    """
    执行全量数据集的科学评估，计算统计指标并绘制图表。
    """
    # 1. 获取全量数据 (不进行 300/25 的筛选限制)
    print("正在扫描 audiolist 文件夹获取全量评估数据...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "audiolist")
    
    audio_paths = []
    labels = []
    
    if not os.path.exists(root_dir):
        print(f"错误：找不到目录 {root_dir}")
        return

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                try:
                    name_parts = os.path.splitext(file)[0].split('_')
                    if len(name_parts) >= 4:
                        v_str = name_parts[-3]
                        a_str = name_parts[-2]
                        
                        if v_str.upper().startswith('V'): v_str = v_str[1:]
                        if a_str.upper().startswith('A'): a_str = a_str[1:]
                        
                        v_val = float(v_str)
                        a_val = float(a_str)
                        
                        audio_paths.append(os.path.join(root, file))
                        labels.append([v_val / 10.0, a_val / 10.0])
                except:
                    continue

    if not audio_paths:
        print("错误：未找到任何符合命名规范的音频数据。")
        return

    print(f"全量评估集共找到 {len(audio_paths)} 首有效歌曲。")

    # 3. 准备数据加载器 (适配新的 MusicDataset 参数)
    dataset = MusicDataset(audio_paths, labels, duration=10.0, train=False)
    # Windows 下 num_workers 建议设为 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 4. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device} 进行推理...")
    
    model = SimpleCNNEmotion().to(device)
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return
    else:
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    model.eval()

    # 5. 批量推理
    all_preds = []
    all_targets = []

    print("正在进行全量预测...")
    from tqdm import tqdm
    with torch.no_grad():
        for mels, targets in tqdm(loader, desc="评估进度"):
            mels = mels.to(device)
            
            # 混合精度推理
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    reg_out, _ = model(mels)
            else:
                reg_out, _ = model(mels)
            
            all_preds.append(reg_out.cpu().numpy())
            all_targets.append(targets.numpy())

    # 拼接结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 分离维度
    v_true = all_targets[:, 0]
    a_true = all_targets[:, 1]
    v_pred = all_preds[:, 0]
    a_pred = all_preds[:, 1]

    # 6. 计算科学指标
    def print_metrics(true, pred, name):
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        p_corr, _ = pearsonr(true, pred)
        
        print(f"\n>>> {name} 评估指标 <<<")
        print(f"{'指标':<20} | {'数值':<10} | {'说明':<30}")
        print("-" * 65)
        print(f"{'MAE (平均绝对误差)':<20} | {mae:.4f}     | 越小越好 (预测值偏离真实值的平均距离)")
        print(f"{'RMSE (均方根误差)':<20} | {rmse:.4f}     | 越小越好 (对大误差更敏感)")
        print(f"{'R² Score (决定系数)':<20} | {r2:.4f}     | 越接近1越好 (模型解释了多少数据方差)")
        print(f"{'Pearson Corr (相关性)':<20} | {p_corr:.4f}     | 越接近1越好 (预测趋势是否一致)")

    print("\n" + "="*65)
    print("📊 深度评估报告")
    print("="*65)
    
    print_metrics(v_true, v_pred, "Valence (愉悦度)")
    print_metrics(a_true, a_pred, "Arousal (兴奋度)")

    # 7. 绘制可视化图表
    print("\n正在绘制可视化图表...")
    plt.style.use('ggplot') # 使用美观的绘图风格
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 通用绘图函数
    def plot_scatter(ax, true, pred, title, color):
        ax.scatter(true, pred, alpha=0.4, s=15, c=color, edgecolors='none')
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal (y=x)')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        
        # 添加相关系数文本
        corr, _ = pearsonr(true, pred)
        ax.text(0.05, 0.9, f'Pearson r = {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))

    def plot_hist(ax, true, pred, title, color):
        error = pred - true
        ax.hist(error, bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', lw=1)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Prediction Error (Pred - True)')
        ax.set_ylabel('Count')
        
        # 添加均值和标准差文本
        mu = np.mean(error)
        sigma = np.std(error)
        # 使用双反斜杠转义 LaTeX 符号，避免 SyntaxWarning
        ax.text(0.05, 0.9, f'$\\mu={mu:.3f}$, $\\sigma={sigma:.3f}$', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

    # 绘制 Valence
    plot_scatter(axes[0, 0], v_true, v_pred, 'Valence: True vs Predicted', 'tab:blue')
    plot_hist(axes[1, 0], v_true, v_pred, 'Valence Error Distribution', 'tab:blue')

    # 绘制 Arousal
    plot_scatter(axes[0, 1], a_true, a_pred, 'Arousal: True vs Predicted', 'tab:green')
    plot_hist(axes[1, 1], a_true, a_pred, 'Arousal Error Distribution', 'tab:green')

    plt.tight_layout()
    save_path = 'evaluation_report_scientific.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 可视化图表已保存至: {os.path.abspath(save_path)}")
    print("请打开该图片查看详细的散点图和误差分布。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="科学评估脚本")
    parser.add_argument("--model", type=str, default="checkpoint_latest.pth", help="模型文件路径")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    args = parser.parse_args()

    # 切换工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    evaluate_scientific(args.model, args.batch_size)
