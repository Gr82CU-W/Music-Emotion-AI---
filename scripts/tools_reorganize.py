import os
import shutil
import re

# 基础路径配置
BASE_DIR = r"d:\music"
AUDIOLIST_DIR = os.path.join(BASE_DIR, "audiolist")

# 情感关键字与文件夹名称的映射
TARGET_FOLDERS = {
    "忧伤": "audio低愉悦_低唤醒_忧伤",
    "愤怒": "audio低愉悦_高唤醒_愤怒",
    "狂欢": "audio高愉悦_高唤醒_狂欢",
    "治愈": "audio高愉悦_低唤醒_治愈"
}

def reorganize_files():
    if not os.path.exists(AUDIOLIST_DIR):
        print(f"目录不存在: {AUDIOLIST_DIR}")
        return

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(AUDIOLIST_DIR) if os.path.isdir(os.path.join(AUDIOLIST_DIR, f))]
    
    move_count = 0
    skip_count = 0

    for subfolder in subfolders:
        current_folder_path = os.path.join(AUDIOLIST_DIR, subfolder)
        files = [f for f in os.listdir(current_folder_path) if f.lower().endswith(".mp3")]
        
        for filename in files:
            file_path = os.path.join(current_folder_path, filename)
            
            # 提取最后一个下划线后面的内容（去掉扩展名）
            # 格式示例: MusicName_V4.50_A5.50_忧伤.mp3
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split("_")
            
            if len(parts) < 2:
                # 没有下划线，无法判断分类，跳过
                continue
                
            predicted_emotion = parts[-1] # 获取最后一个部分
            
            if predicted_emotion in TARGET_FOLDERS:
                target_subfolder = TARGET_FOLDERS[predicted_emotion]
                target_folder_path = os.path.join(AUDIOLIST_DIR, target_subfolder)
                
                # 确保目标文件夹存在
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)
                
                # 如果当前文件夹不是目标文件夹，则移动
                if subfolder != target_subfolder:
                    dst_path = os.path.join(target_folder_path, filename)
                    
                    # 处理重名冲突
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(filename)
                        dst_path = os.path.join(target_folder_path, f"{base}_moved{ext}")
                    
                    try:
                        shutil.move(file_path, dst_path)
                        print(f"移动: {filename} \n  从: {subfolder} \n  到: {target_subfolder}")
                        move_count += 1
                    except Exception as e:
                        print(f"移动失败 {filename}: {e}")
                else:
                    skip_count += 1
            else:
                # 最后一个下划线后不是已知的情感标签，跳过
                pass

    print(f"\n整理完成！")
    print(f"成功移动文件: {move_count} 个")
    print(f"已在正确位置的文件: {skip_count} 个")

if __name__ == "__main__":
    reorganize_files()
