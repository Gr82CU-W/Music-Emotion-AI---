import os
import time
import re
import librosa
import numpy as np
from playwright.sync_api import sync_playwright

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIOLIST_DIR = os.path.join(BASE_DIR, "audiolist")
URL = "http://mtg.github.io/essentia.js/examples/#/demos/"

def get_all_mp3_files(directory):
    """递归获取目录下所有 mp3 文件"""
    mp3_files = []
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return mp3_files
        
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp3"):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

def check_audio_validity(file_path):
    """
    尝试加载音频文件的前 0.5 秒，检测文件是否损坏。
    """
    try:
        # 使用 librosa 尝试加载，这是最可靠的方法
        # sr=None 保持原始采样率，duration=0.5 只读取前0.5秒
        librosa.load(file_path, sr=None, duration=0.5)
        return True
    except Exception as e:
        print(f"  -> [警告] 音频文件损坏或无法读取: {os.path.basename(file_path)} ({e})")
        return False

def get_emotion_category(valence, arousal):
    """
    根据 Valence 和 Arousal (0-10 范围) 判断情感分类。
    A<5 V<5 -> 忧伤 (Low V, Low A)
    A>5 V>5 -> 狂欢 (High V, High A)
    A<5 V>5 -> 治愈 (High V, Low A)
    A>5 V<5 -> 愤怒 (Low V, High A)
    """
    # 注意：这里使用了简单的 >= 5 和 < 5 的划分
    if arousal < 5 and valence < 5:
        return "忧伤"
    elif arousal >= 5 and valence >= 5:
        return "狂欢"
    elif arousal < 5 and valence >= 5:
        return "治愈"
    elif arousal >= 5 and valence < 5:
        return "愤怒"
    else:
        # 理论上不会走到这里，除非有 NaN
        return "未知"

def update_filename(file_path, valence_raw, arousal_raw):
    """
    更新文件名。
    1. 提取第一个下划线前的部分作为歌名。
    2. 将原始 V/A (0-1) 缩放到 0-10。
    3. 根据缩放后的值确定情感分类。
    4. 重命名文件。
    """
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 1. 提取歌名 (第一个下划线前的内容)
    if "_" in name_without_ext:
        music_name = name_without_ext.split("_")[0]
    else:
        music_name = name_without_ext
        
    # 2. 使用原始数值 (网页返回的已经是 0-10 范围)
    # 仅做 0-10 的边界限制，不做倍数缩放
    v_scaled = min(max(valence_raw, 0), 10)
    a_scaled = min(max(arousal_raw, 0), 10)
    
    # 3. 获取情感分类
    emotion = get_emotion_category(v_scaled, a_scaled)
    
    # 4. 构建新文件名
    # 格式: 名字_V{:.2f}_A{:.2f}_{情感}.mp3
    new_filename = f"{music_name}_V{v_scaled:.2f}_A{a_scaled:.2f}_{emotion}.mp3"
    new_file_path = os.path.join(directory, new_filename)
    
    try:
        if new_file_path != file_path:
            os.rename(file_path, new_file_path)
            print(f"  -> 文件已重命名:\n     旧: {filename}\n     新: {new_filename}")
        else:
            print(f"  -> 文件名无需更新: {filename}")
        return new_file_path
    except Exception as e:
        print(f"  -> 重命名失败: {e}")
        return file_path

def run_batch_process():
    files = get_all_mp3_files(AUDIOLIST_DIR)
    print(f"在 {AUDIOLIST_DIR} 下找到 {len(files)} 个 MP3 文件。")

    if not files:
        return

def process_files_with_browser(files, progress_callback=None):
    """
    使用 Playwright 处理文件列表。
    progress_callback(current_index, total, filename, result_dict)
    returns: list of results [{"path": str, "valence": float, "arousal": float}]
    """
    results = []
    if not files:
        return results

    with sync_playwright() as p:
        # headless=True 以避免干扰，如果需要演示可以设为 False
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()

        # 用于存储当前文件的结果
        current_processing_result = {}
        
        def handle_console(msg):
            if msg.type != "log":
                return
            for arg in msg.args:
                try:
                    val = arg.json_value()
                    if isinstance(val, dict) and "valence" in val and "arousal" in val:
                        current_processing_result["valence"] = float(val["valence"])
                        current_processing_result["arousal"] = float(val["arousal"])
                except:
                    pass

        page.on("console", handle_console)

        # 预加载页面
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)
        except Exception as e:
            print(f"页面加载出错: {e}")
            return results

        for index, file_path in enumerate(files):
            # 转换为绝对路径以确保 Playwright 能够找到文件
            if not os.path.isabs(file_path):
                abs_file_path = os.path.abspath(file_path)
            else:
                abs_file_path = file_path

            filename = os.path.basename(abs_file_path)
            print(f"[{index+1}/{len(files)}] 正在处理: {filename}")
            
            if progress_callback:
                progress_callback(index, len(files), filename, None)

            # 1. 检测音频是否有效
            if not check_audio_validity(abs_file_path):
                print("  -> 跳过: 音频无效")
                continue

            current_processing_result.clear()
            
            try:
                # 刷新页面或直接上传 (页面如果状态没清空可能需要刷新，但频繁刷新慢，这里尝试直接上传)
                # 为了稳定性，每次刷新一下比较好，或者利用SPA特性
                # 简单起见，每几个刷新一次，或者每次刷新
                # 如果不刷新，上一个结果可能残留？current_processing_result已清空。
                # 但是页面上的显示可能残留。
                # 安全起见，刷新。
                try:
                    page.reload(wait_until="domcontentloaded", timeout=10000)
                except:
                    pass
                page.wait_for_timeout(1000)

                # 上传文件
                page.locator("#file-select-area input[type='file']").set_input_files(abs_file_path)

                # 等待结果
                wait_start = time.time()
                got_result = False
                while time.time() - wait_start < 30:
                    if "valence" in current_processing_result:
                        got_result = True
                        break
                    page.wait_for_timeout(500)
                
                if got_result:
                    v = current_processing_result["valence"]
                    a = current_processing_result["arousal"]
                    # 原始是 0-10
                    # 我们这里返回原始值，并在外部处理
                    res = {"path": file_path, "valence": v, "arousal": a}
                    results.append(res)
                    if progress_callback:
                        progress_callback(index+1, len(files), filename, res)
                    
                    # 也可以选择在这里调用 update_filename
                    # update_filename(file_path, v, a) 
                    # 但让调用者决定
                else:
                    print("  -> 超时: 未获取到结果")

            except Exception as e:
                print(f"  -> 处理出错: {e}")
                continue
        
        browser.close()
    return results

def run_batch_process():
    files = get_all_mp3_files(AUDIOLIST_DIR)
    print(f"在 {AUDIOLIST_DIR} 下找到 {len(files)} 个 MP3 文件。")

    # 复用新的处理函数
    def console_log_callback(idx, total, fname, res):
        if res:
            print(f"  -> V={res['valence']:.4f}, A={res['arousal']:.4f}")
            update_filename(res['path'], res['valence'], res['arousal'])
    
    process_files_with_browser(files, progress_callback=console_log_callback)


if __name__ == "__main__":
    run_batch_process()
