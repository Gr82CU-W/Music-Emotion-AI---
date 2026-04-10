import torch
from transformers import AutoTokenizer
from models.va_regressor import VARegressor
import numpy as np
from langdetect import detect
import re
import os

# 模型和tokenizer路径
# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EN_MODEL_PATH = os.path.join(BASE_DIR, "pretrained/bert-base-uncased")
EN_MODEL_FILE = os.path.join(BASE_DIR, "va_model.pt")
ZH_MODEL_PATH = os.path.join(BASE_DIR, "pretrained/chinese-roberta-wwm-ext")
ZH_MODEL_FILE = os.path.join(BASE_DIR, "zh_va_model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载英文模型和tokenizer
en_tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_PATH, local_files_only=True)
en_model = VARegressor(EN_MODEL_PATH).to(device)
en_model.load_state_dict(torch.load(EN_MODEL_FILE, map_location=device))
en_model.eval()

# 加载中文模型和tokenizer
zh_tokenizer = AutoTokenizer.from_pretrained(ZH_MODEL_PATH, local_files_only=True)
zh_model = VARegressor(ZH_MODEL_PATH).to(device)
zh_model.load_state_dict(torch.load(ZH_MODEL_FILE, map_location=device))
zh_model.eval()

def detect_language(text):
    """
    检测文本语言：中文或英文
    """
    # 简单检查：如果包含中文字符，则为中文
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    try:
        lang = detect(text)
        if lang == "zh-cn" or lang == "zh":
            return "zh"
        elif lang == "en":
            return "en"
        else:
            # 默认英文
            return "en"
    except:
        # 检测失败，默认英文
        return "en"

def predict_va(text):
    """
    预测文本的valence和arousal值
    """
    lang = detect_language(text)

    if lang == "zh":
        tokenizer = zh_tokenizer
        model = zh_model
        max_len = 128  # 根据中文数据集调整
    else:
        tokenizer = en_tokenizer
        model = en_model
        max_len = 128

    # 预处理文本
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        v_pred, a_pred = model(input_ids, attention_mask)

    # 反归一化
    if lang == "zh":
        # 中文数据集的归一化：假设valence和arousal在[0,1]或类似
        # 从train_zh.py看，valence和arousal可能是原始值或归一化的
        # 假设是原始值范围，需要根据数据集调整
        v = v_pred.item()
        a = a_pred.item()
    else:
        # 英文数据集：valence [-1,1] -> [1,5], arousal [0,1] -> [1,5]
        v = v_pred.item()   # [-1,1] -> [1,5]
        a = a_pred.item()   # [0,1] -> [1,5]

    return {"valence": round(v, 4), "arousal": round(a, 4), "language": lang}

if __name__ == "__main__":
    # 示例使用
    texts = [
        "I am very happy today!",
        "今天天气真好，我很开心！",
        "This is an amazing experience.",
        "这是一个令人惊叹的经历。",
        "我感到很幸福。",
        "I feel so happy."
    ]

    for text in texts:
        result = predict_va(text)
        print(f"Text: {text}")
        print(f"Language: {result['language']}")
        print(f"Valence: {result['valence']}, Arousal: {result['arousal']}")
        print("-" * 50)