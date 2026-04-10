import torch
import librosa
import numpy as np
import os
import sys

# 添加项目根目录到 sys.path，以便能够导入 core 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.model_audio_cnn import SimpleCNNEmotion, extract_features

def predict_emotion(model, audio_path, device, sr=22050, duration=10.0):
    """
    对单个音频文件进行情感预测
    """
    target_len = int(sr * duration)
    try:
        # 1. 加载音频
        # 使用 duration 参数可以加快读取速度，但为了保持与训练一致（中心裁剪），我们先加载全部或较长一段
        # 如果音频很长，librosa.load 可能会比较慢。
        y, _ = librosa.load(audio_path, sr=sr)
        
        # 2. 中心裁剪 (与 MusicDataset 一致)
        if len(y) > target_len:
            center = len(y) // 2
            start = center - (target_len // 2)
            y = y[start : start + target_len]
        elif len(y) < target_len:
            y = librosa.util.fix_length(y, size=target_len)
            
        # 3. 提取特征 (3通道: Mel + Delta + Delta2)
        features = extract_features(y, sr=sr, n_mels=128)
        
        # 4. 尺寸固定 (10秒约 430 帧)
        target_frames = 430
        features = librosa.util.fix_length(features, size=target_frames, axis=2)
        
        # 5. 转换为 tensor 并推理
        # features shape: (3, 128, 430) -> (1, 3, 128, 430)
        input_tensor = torch.tensor(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            # 假设输出是 [Valence, Arousal]
            v, a = output[0].cpu().numpy()
            
        return v, a
    except Exception as e:
        print(f"跳过文件 {os.path.basename(audio_path)}: 无法读取或处理 ({e})")
        return None, None

def get_emotion_category(v, a):
    """
    根据 Valence 和 Arousal 的预测值判断情感分类
    阈值设为 0.5
    """
    if v > 0.5:
        if a > 0.5:
            return "狂欢"
        else:
            return "治愈"
    else:
        if a > 0.5:
            return "愤怒"
        else:
            return "忧伤"

def main():
    # 1. 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "emotion_model.pth")
    audiolist_dir = os.path.join(current_dir, "audiolist")
    
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    if not os.path.exists(audiolist_dir):
        print(f"错误：找不到音频目录 {audiolist_dir}")
        return

    # 2. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = SimpleCNNEmotion(emotion_dim=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("模型加载成功。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 遍历 audiolist 下的子文件夹
    subfolders = [f for f in os.listdir(audiolist_dir) if os.path.isdir(os.path.join(audiolist_dir, f))]
    
    for folder in subfolders:
        folder_path = os.path.join(audiolist_dir, folder)
        print(f"\n正在处理文件夹: {folder}")
        
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        for filename in files:
            # 跳过非音频文件（简单判断后缀）
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.mp3', '.wav', '.flac', '.m4a', '.mps']: # 包含用户提到的 .mps
                continue
                
            # 如果文件名已经包含预测结果（防止重复运行），可以根据需要决定是否跳过
            if "_V" in filename and "_A" in filename:
                print(f"跳过已处理文件: {filename}")
                continue

            file_path = os.path.join(folder_path, filename)
            
            # 预测
            v, a = predict_emotion(model, file_path, device)
            
            if v is not None and a is not None:
                category = get_emotion_category(v, a)
                
                # 构建新文件名
                # 格式: 原文件名_V0.xx_A0.xx_类别.后缀
                name_part, extension = os.path.splitext(filename)
                new_filename = f"{name_part}_V{v:.2f}_A{a:.2f}_{category}{extension}"
                new_file_path = os.path.join(folder_path, new_filename)
                
                try:
                    os.rename(file_path, new_file_path)
                    print(f"已重命名: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"重命名失败 {filename}: {e}")

    print("\n所有处理已完成。")

if __name__ == "__main__":
    main()
