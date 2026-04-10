import os
import pandas as pd
import shutil
import numpy as np

# 路径配置
BASE_DIR = r"d:\music"
PMEMO_DIR = os.path.join(BASE_DIR, "PMEmo2019", "PMEmo2019")
PMEMO_ANNOTATION_FILE = os.path.join(PMEMO_DIR, "annotations", "static_annotations.csv")
PMEMO_METADATA_FILE = os.path.join(PMEMO_DIR, "metadata.csv")
PMEMO_AUDIO_SRC_DIR = os.path.join(PMEMO_DIR, "chorus")

DEAM_DIR = os.path.join(BASE_DIR, "DEAM")
DEAM_ANNOTATION_FILES = [
    os.path.join(DEAM_DIR, "annotations", "annotations averaged per song", "song_level", "static_annotations_averaged_songs_1_2000.csv"),
    os.path.join(DEAM_DIR, "annotations", "annotations averaged per song", "song_level", "static_annotations_averaged_songs_2000_2058.csv")
]
DEAM_AUDIO_SRC_DIR = os.path.join(DEAM_DIR, "MEMD_audio")

AUDIOLIST_DIR = os.path.join(BASE_DIR, "audiolist")

# 目标文件夹映射
TARGET_FOLDERS = {
    "忧伤": "audio低愉悦_低唤醒_忧伤",
    "愤怒": "audio低愉悦_高唤醒_愤怒",
    "狂欢": "audio高愉悦_高唤醒_狂欢",
    "治愈": "audio高愉悦_低唤醒_治愈"
}

def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    return "".join([c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')]).strip()

def get_existing_music_names():
    """获取四个目标文件夹中已经存在的音乐名称（不含后缀）"""
    existing_names = set()
    for folder in TARGET_FOLDERS.values():
        folder_path = os.path.join(AUDIOLIST_DIR, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".mp3"):
                    name = os.path.splitext(f)[0]
                    # 如果有 _id 后缀，尝试还原
                    if "_" in name:
                        # 检查最后一部分是否是数字
                        parts = name.rsplit("_", 1)
                        if parts[1].isdigit():
                            existing_names.add(parts[0])
                    existing_names.add(name)
    return existing_names

def import_all_music():
    existing_names = get_existing_music_names()
    print(f"当前已存在 {len(existing_names)} 首音乐。")

    # 1. 处理 PMEmo 数据
    print("\n正在处理 PMEmo 数据...")
    if os.path.exists(PMEMO_ANNOTATION_FILE) and os.path.exists(PMEMO_METADATA_FILE):
        df_ann = pd.read_csv(PMEMO_ANNOTATION_FILE)
        df_meta = pd.read_csv(PMEMO_METADATA_FILE)
        df_pmemo = pd.merge(df_ann, df_meta[['musicId', 'title']], on='musicId')
        
        count_pmemo = 0
        for _, row in df_pmemo.iterrows():
            music_id = int(row["musicId"])
            title = str(row["title"])
            v = float(row["Valence(mean)"])
            a = float(row["Arousal(mean)"])
            
            safe_title = sanitize_filename(title)
            if not safe_title:
                safe_title = str(music_id)
            
            if safe_title in existing_names:
                continue
                
            # 分类
            if v >= 0.5 and a >= 0.5: cat = "狂欢"
            elif v < 0.5 and a >= 0.5: cat = "愤怒"
            elif v < 0.5 and a < 0.5: cat = "忧伤"
            else: cat = "治愈"
            
            target_path = os.path.join(AUDIOLIST_DIR, TARGET_FOLDERS[cat])
            if not os.path.exists(target_path):
                os.makedirs(target_path)
                
            src_file = os.path.join(PMEMO_AUDIO_SRC_DIR, f"{music_id}.mp3")
            dst_file = os.path.join(target_path, f"{safe_title}.mp3")
            
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dst_file)
                    existing_names.add(safe_title)
                    count_pmemo += 1
                except Exception as e:
                    print(f"  -> PMEmo 复制失败 {music_id}: {e}")
        print(f"PMEmo 处理完成，导入了 {count_pmemo} 首。")
    else:
        print("找不到 PMEmo 标注文件或元数据文件。")

    # 2. 处理 DEAM 数据
    print("\n正在处理 DEAM 数据...")
    count_deam = 0
    for csv_path in DEAM_ANNOTATION_FILES:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                song_id = int(row["song_id"])
                # DEAM 没有 metadata，直接用 ID 作为名字
                title = str(song_id)
                
                if title in existing_names:
                    continue
                
                # 归一化 DEAM 的 VA 值 (1-9 -> 0-1)
                v = (float(row['valence_mean']) - 1.0) / 8.0
                a = (float(row['arousal_mean']) - 1.0) / 8.0
                v = max(0, min(1, v))
                a = max(0, min(1, a))
                
                # 分类
                if v >= 0.5 and a >= 0.5: cat = "狂欢"
                elif v < 0.5 and a >= 0.5: cat = "愤怒"
                elif v < 0.5 and a < 0.5: cat = "忧伤"
                else: cat = "治愈"
                
                target_path = os.path.join(AUDIOLIST_DIR, TARGET_FOLDERS[cat])
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                    
                src_file = os.path.join(DEAM_AUDIO_SRC_DIR, f"{song_id}.mp3")
                dst_file = os.path.join(target_path, f"{title}.mp3")
                
                if os.path.exists(src_file):
                    try:
                        shutil.copy2(src_file, dst_file)
                        existing_names.add(title)
                        count_deam += 1
                    except Exception as e:
                        print(f"  -> DEAM 复制失败 {song_id}: {e}")
        else:
            print(f"找不到 DEAM 标注文件: {csv_path}")
    print(f"DEAM 处理完成，导入了 {count_deam} 首。")

if __name__ == "__main__":
    import_all_music()
