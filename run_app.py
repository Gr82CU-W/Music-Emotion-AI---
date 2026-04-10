import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import os
import sys
import json
import time

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 设置 Matplotlib 中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# Ensure local imports work
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'predict_va'))

# Try importing modules
try:
    from predict_va.predict_combined import predict_va
    PREDICT_VA_AVAILABLE = True
except ImportError:
    # print("Warning: predict_va module not found.")
    PREDICT_VA_AVAILABLE = False

try:
    from core.engine_audio_va import process_files_with_browser
    GET_VA_AVAILABLE = True
except ImportError:
    # print("Warning: getVA module failed.")
    GET_VA_AVAILABLE = False

USER_DATA_DIR = "user_data"
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

class ModernStyle:
    BG_COLOR = "#f5f7fa"       # 浅灰背景
    CARD_COLOR = "#ffffff"     # 白色卡片
    ACCENT_COLOR = "#3498db"   # 科技蓝
    TEXT_COLOR = "#2c3e50"     # 深灰文字
    SUBTEXT_COLOR = "#7f8c8d"  # 辅助文字
    SUCCESS_COLOR = "#2ecc71"  # 绿色
    
    FONT_TITLE = ("微软雅黑", 16, "bold")
    FONT_SUBTITLE = ("微软雅黑", 12, "bold")
    FONT_BODY = ("微软雅黑", 10)
    FONT_SMALL = ("微软雅黑", 9)

    @staticmethod
    def apply_style():
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame
        style.configure("Card.TFrame", background=ModernStyle.CARD_COLOR, relief="flat")
        style.configure("Main.TFrame", background=ModernStyle.BG_COLOR)
        
        # Label
        style.configure("TLabel", background=ModernStyle.CARD_COLOR, foreground=ModernStyle.TEXT_COLOR, font=ModernStyle.FONT_BODY)
        style.configure("Title.TLabel", font=ModernStyle.FONT_TITLE, foreground=ModernStyle.ACCENT_COLOR, background=ModernStyle.CARD_COLOR)
        style.configure("Subtitle.TLabel", font=ModernStyle.FONT_SUBTITLE, foreground=ModernStyle.TEXT_COLOR, background=ModernStyle.CARD_COLOR)
        style.configure("Info.TLabel", font=ModernStyle.FONT_SMALL, foreground=ModernStyle.SUBTEXT_COLOR, background=ModernStyle.CARD_COLOR)
        style.configure("MainInfo.TLabel", font=ModernStyle.FONT_SMALL, foreground=ModernStyle.SUBTEXT_COLOR, background=ModernStyle.BG_COLOR)
        
        # Button
        style.configure("TButton", font=("微软雅黑", 10), padding=4, background=ModernStyle.ACCENT_COLOR, foreground="white", borderwidth=0)
        style.map("TButton", 
                  background=[('pressed', '#1f6391'), ('active', '#2980b9'), ('disabled', '#bdc3c7')],
                  foreground=[('!disabled', 'white'), ('disabled', '#a5a5a5')])
        
        style.configure("Action.TButton", font=("微软雅黑", 10, "bold"), padding=5, background=ModernStyle.ACCENT_COLOR, foreground="white")
        style.map("Action.TButton", 
                  background=[('pressed', '#1f6391'), ('active', '#2980b9')],
                  foreground=[('active', 'white'), ('!disabled', 'white')])
        
        style.configure("Outline.TButton", background="#ecf0f1", foreground=ModernStyle.TEXT_COLOR)

        # Radiobutton
        style.configure("TRadiobutton", background=ModernStyle.CARD_COLOR, font=ModernStyle.FONT_BODY, foreground=ModernStyle.TEXT_COLOR)

class LoginWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Music Emotion AI - 系统登录")
        self.root.geometry("500x450")
        self.root.configure(bg=ModernStyle.BG_COLOR)
        
        ModernStyle.apply_style()
        self.center_window(500, 450)

        # Container with padding to center the card
        container = tk.Frame(self.root, bg=ModernStyle.BG_COLOR)
        container.pack(expand=True, fill='both', padx=30, pady=30)

        # Main Card using pack instead of place for better auto-sizing
        card = ttk.Frame(container, style="Card.TFrame", padding=30)
        card.pack(expand=True, fill='both')

        # Title
        ttk.Label(card, text="Music Emotion AI", font=("Arial", 26, "bold"), foreground=ModernStyle.ACCENT_COLOR).pack(pady=(10, 5))
        ttk.Label(card, text="情绪感知与音乐推荐系统", font=("微软雅黑", 13), foreground=ModernStyle.SUBTEXT_COLOR).pack(pady=(0, 30))

        # Inputs
        input_frame = ttk.Frame(card, style="Card.TFrame")
        input_frame.pack(fill='x', padx=10)

        ttk.Label(input_frame, text="账号 (Account)", font=ModernStyle.FONT_SMALL).pack(anchor='w')
        self.entry_user = ttk.Entry(input_frame, font=("微软雅黑", 12))
        self.entry_user.pack(fill='x', pady=(5, 15))

        ttk.Label(input_frame, text="密码 (Password)", font=ModernStyle.FONT_SMALL).pack(anchor='w')
        self.entry_pass = ttk.Entry(input_frame, show="●", font=("微软雅黑", 12))
        self.entry_pass.pack(fill='x', pady=(5, 25))

        # Button - Explicitly adding a frame for the button to control its visibility
        btn_frame = ttk.Frame(card, style="Card.TFrame")
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        self.btn_login = ttk.Button(btn_frame, text="立即登录 / 注册新账号", style="Action.TButton", command=self.login)
        self.btn_login.pack(fill='x')
        
        ttk.Label(card, text="* 首次开启系统将自动为您同步注册信息", style="Info.TLabel").pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def center_window(self, width, height):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f'{width}x{height}+{(sw-width)//2}+{(sh-height)//2}')

    def login(self):
        username = self.entry_user.get().strip()
        password = self.entry_pass.get().strip()
        
        if not username or not password:
            messagebox.showwarning("提示", "请输入完整的账号密码")
            return

        user_file = os.path.join(USER_DATA_DIR, "users.json")
        users = {}
        if os.path.exists(user_file):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    users = json.load(f)
            except:
                pass
        
        if username in users:
            if users[username] == password:
                self.launch_main(username)
            else:
                messagebox.showerror("登录失败", "密码错误，请重试。")
        else:
            users[username] = password
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(users, f)
            messagebox.showinfo("注册成功", f"欢迎新用户 {username}，正在初始化环境...")
            self.launch_main(username)

    def launch_main(self, username):
        self.root.destroy()
        root = tk.Tk()
        app = MainWindow(root, username)
        root.mainloop()

    def on_close(self):
        self.root.destroy()
        sys.exit()

class UploadWindow:
    def __init__(self, parent, callback_add_data, username):
        self.top = tk.Toplevel(parent)
        self.top.title("上传音乐 - 专属数据库扩充")
        self.top.geometry("900x700")
        self.top.configure(bg=ModernStyle.BG_COLOR)
        self.top.attributes("-topmost", True) # 窗口置顶 (User Request)
        
        self.callback = callback_add_data
        self.username = username
        self.pending_files = [] # list of paths
        
        # Layout
        container = ttk.Frame(self.top, style="Main.TFrame", padding=20)
        container.pack(fill='both', expand=True)

        # Header
        info_frame = ttk.Frame(container, style="Main.TFrame")
        info_frame.pack(fill='x', pady=(0, 15))
        ttk.Label(info_frame, text="音乐特征提取工坊 (Music Feature Extraction Lab)", style="Subtitle.TLabel", background=ModernStyle.BG_COLOR).pack(anchor='w')
        
        desc = ("【核心功能说明】\n"
                "1. 本模块用于构建您的【私有音乐情绪数据库】。\n"
                "2. 导入的 MP3 文件将被送入后端 AI 引擎，通过 Playwright 模拟人类听感分析 Valence (愉悦度) 和 Arousal (唤醒度)。\n"
                "3. 分析结果将永久保存至您的账户下，不仅用于本次推荐，也会不断训练您的个人偏好模型。")
        ttk.Label(info_frame, text=desc, style="MainInfo.TLabel", wraplength=850, justify='left').pack(anchor='w', pady=5)

        # Toolbar
        btn_frame = ttk.Frame(container, style="Main.TFrame")
        btn_frame.pack(fill='x', pady=(0, 10))
        
        # Consistent height with tk.Button
        tk.Button(btn_frame, text="📄 添加文件 (Add Files)", bg="#ecf0f1", relief="flat", padx=10, command=self.add_files).pack(side='left', padx=(0, 10))
        tk.Button(btn_frame, text="📂 添加文件夹 (Add Folder)", bg="#ecf0f1", relief="flat", padx=10, command=self.add_folder).pack(side='left', padx=(0, 10))
        
        # Clear button with higher contrast (User Request: 清空列表按键灰色太浅)
        tk.Button(btn_frame, text="🗑 清空列表", bg="#95a5a6", fg="white", relief="flat", padx=10, command=self.clear_list).pack(side='left', padx=(0, 10))
        
        # Using tk.Button for high visibility
        self.btn_process = tk.Button(btn_frame, text="🚀 批量启动分析引擎 (Process All)", 
                                     font=("微软雅黑", 10, "bold"),
                                     bg=ModernStyle.SUCCESS_COLOR,
                                     fg="white",
                                     relief="flat",
                                     padx=15,
                                     state='disabled',
                                     command=self.start_process)
        self.btn_process.pack(side='right')
        
        # File List
        tree_frame = ttk.Frame(container)
        tree_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        cols = ("file", "path", "status", "result")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)
        self.tree.heading("file", text="音乐文件名")
        self.tree.heading("path", text="完整路径")
        self.tree.heading("status", text="处理状态")
        self.tree.heading("result", text="分析结果 (V, A)")
        
        self.tree.column("file", width=200)
        self.tree.column("path", width=300)
        self.tree.column("status", width=100, anchor='center')
        self.tree.column("result", width=150, anchor='center')
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        # Log
        log_frame = ttk.Labelframe(container, text="系统内核日志", padding=10)
        log_frame.pack(fill='x')
        self.log_text = tk.Text(log_frame, height=5, font=("Consolas", 9), bg="#fafafa", relief="flat", state='disabled')
        self.log_text.pack(fill='x')

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("MP3 Files", "*.mp3")], parent=self.top)
        if files:
            self._append_files(files)

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self.top)
        if folder:
            mp3s = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith('.mp3'):
                        mp3s.append(os.path.join(root, f))
            if mp3s:
                self._append_files(mp3s)
            else:
                messagebox.showinfo("提示", "该文件夹下未找到MP3文件", parent=self.top)

    def _append_files(self, file_list):
        count = 0
        cwd = os.getcwd()
        for f in file_list:
            # 转换为相对路径 (相对于项目根目录)
            try:
                rel_f = os.path.relpath(f, cwd)
                # 如果不在项目目录下或者是跨盘符，则保留绝对路径
                if rel_f.startswith('..') or (os.path.isabs(rel_f) and not rel_f.startswith(cwd)):
                    final_path = f
                else:
                    final_path = rel_f
            except ValueError:
                final_path = f

            if final_path not in self.pending_files:
                self.pending_files.append(final_path)
                fname = os.path.basename(final_path)
                self.tree.insert("", "end", iid=final_path, values=(fname, final_path, "等待中", "-"))
                count += 1
        
        if count > 0:
            self.btn_process['state'] = 'normal'
            self.log(f"已添加 {count} 首新歌到任务队列。当前总计: {len(self.pending_files)}")
            
    def clear_list(self):
        self.pending_files.clear()
        self.tree.delete(*self.tree.get_children())
        self.btn_process['state'] = 'disabled'
        self.log("列表已清空。")

    def start_process(self):
        if not GET_VA_AVAILABLE:
            messagebox.showerror("组件缺失", "无法调用 getVA 模块，请检查环境。", parent=self.top)
            return
        
        # 允许重复处理（有些可能之前失败了），或者只处理等待中的
        files_to_process = [f for f in self.pending_files if self.tree.item(f)['values'][2] != "✅ 完成"]
        
        if not files_to_process:
            messagebox.showinfo("提示", "当前列表中没有需要分析的任务。", parent=self.top)
            return

        self.btn_process['state'] = 'disabled'
        self.log(f"启动分析引擎... 队列长度: {len(files_to_process)}")
        
        def run_task():
            processor = ProcessThread(files_to_process, self.update_row, self.log)
            # 这里的 run 会阻塞直到全部处理完或出错
            all_results = processor.run()
            
            # 无论是否有结果，都回到主线程进行收尾，确保界面响应
            def finalize():
                try:
                    if all_results:
                        # 1. 明确尝试保存数据并获取返回值
                        save_success = self.callback(all_results, self.username)
                        
                        if save_success:
                            self.log(f"成功保存到私有库：{len(all_results)} 首")
                            self.top.attributes("-topmost", False)
                            messagebox.showinfo("处理成功", f"恭喜！{len(all_results)} 首音乐特征已安全存入您的私有库。\n数据已并入主界面推荐引擎。", parent=self.top)
                            self.top.attributes("-topmost", True)
                        else:
                            self.log("⚠️ 数据保存失败，请检查硬盘写入权限。")
                    else:
                        self.top.attributes("-topmost", False)
                        messagebox.showwarning("无结果", "分析完成，但未提取到任何有效 V-A 特征。可能是音频格式不支持或网络环境受限。", parent=self.top)
                        self.top.attributes("-topmost", True)
                except Exception as e:
                    self.log(f"收尾异常: {str(e)}")
                finally:
                    self.btn_process.config(state='normal')

            self.top.after(0, finalize)

        threading.Thread(target=run_task, daemon=True).start()

    def update_row(self, filepath, status, v="-", a="-"):
        try:
            if self.tree.exists(filepath):
                current_values = self.tree.item(filepath)['values']
                # Keep filename and path, update status and result
                self.tree.item(filepath, values=(current_values[0], current_values[1], status, f"V:{v}, A:{a}" if v != "-" else "-"))
                self.tree.see(filepath)
        except:
            pass

class ManageWindow:
    def __init__(self, parent, username, refresh_callback):
        self.top = tk.Toplevel(parent)
        self.top.title(f"私有库管理器 - {username}")
        self.top.geometry("800x600")
        self.top.configure(bg=ModernStyle.BG_COLOR)
        self.top.attributes("-topmost", True)
        
        self.username = username
        self.refresh_callback = refresh_callback
        self.user_file = os.path.join(USER_DATA_DIR, f"{username}_dataset.csv")
        
        # Layout
        container = ttk.Frame(self.top, style="Main.TFrame", padding=20)
        container.pack(fill='both', expand=True)
        
        ttk.Label(container, text="📂 现有私有库内容列表", style="Subtitle.TLabel", background=ModernStyle.BG_COLOR).pack(anchor='w', pady=(0, 10))
        
        # Toolbar
        tbar = ttk.Frame(container, style="Main.TFrame")
        tbar.pack(fill='x', pady=(0, 10))
        tk.Button(tbar, text="🗑 删除选中曲目", bg="#e74c3c", fg="white", relief="flat", padx=10, command=self.delete_selected).pack(side='left', padx=5)
        tk.Button(tbar, text="💾 保存并刷新主界面", bg=ModernStyle.SUCCESS_COLOR, fg="white", relief="flat", padx=10, command=self.save_and_exit).pack(side='right', padx=5)

        # Treeview
        cols = ("file", "v", "a", "path")
        self.tree = ttk.Treeview(container, columns=cols, show="headings")
        self.tree.heading("file", text="文件名")
        self.tree.heading("v", text="Valence")
        self.tree.heading("a", text="Arousal")
        self.tree.heading("path", text="物理路径")
        
        self.tree.column("file", width=200)
        self.tree.column("v", width=80, anchor='center')
        self.tree.column("a", width=80, anchor='center')
        self.tree.column("path", width=300)
        
        self.tree.pack(fill='both', expand=True)
        
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.user_file):
            return
        try:
            df = pd.read_csv(self.user_file)
            for i, row in df.iterrows():
                self.tree.insert("", "end", values=(row['filename'], f"{row['v']:.3f}", f"{row['a']:.3f}", row['path']))
        except:
            pass

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel: return
        if messagebox.askyesno("确认", "确定要从私有库中删除选中项吗？(本地文件不会被删除)", parent=self.top):
            for item in sel:
                self.tree.delete(item)

    def save_and_exit(self):
        # Collect from tree
        data = []
        for item in self.tree.get_children():
            v = self.tree.item(item)['values']
            data.append({'filename': v[0], 'v': float(v[1]), 'a': float(v[2]), 'path': v[3]})
        
        new_df = pd.DataFrame(data)
        new_df.to_csv(self.user_file, index=False)
        self.refresh_callback()
        messagebox.showinfo("成功", "私有库已更新！", parent=self.top)
        self.top.destroy()

class ProcessThread:
    def __init__(self, files, update_cb, log_cb):
        self.files = files
        self.update_cb = update_cb
        self.log_cb = log_cb
        self.results = []

    def run(self):
        def internal_cb(idx, total, fname, result):
            if result:
                # result 包含 {'path', 'valence', 'arousal'}
                self.update_cb(result['path'], "✅ 完成", f"{result['valence']:.3f}", f"{result['arousal']:.3f}")
                
                # 统一 key 为 'v' 和 'a' 适配主界面的逻辑
                self.results.append({
                    'path': result['path'],
                    'filename': fname,
                    'v': result['valence'],
                    'a': result['arousal']
                })
            else:
                self.log_cb(f"进程 [{idx}/{total}] 正在提取特征: {fname}")
        
        try:
            self.log_cb("正在初始化 Playwright 虚拟浏览器环境...")
            # process_files_with_browser 内部会调用 internal_cb
            process_files_with_browser(self.files, progress_callback=internal_cb)
            
            self.log_cb(f"分析线程结束，共获取 {len(self.results)} 条有效数据。")
            return self.results
        except Exception as e:
            self.log_cb(f"后端引擎严重故障: {str(e)}")
            return []

class MainWindow:
    def __init__(self, root, username):
        self.root = root
        self.username = username
        self.root.title(f"Music Emotion AI - {username} (Console Mode)")
        
        # 针对 2560x1600 屏幕进行极限适配
        # 设置宽大的初始尺寸
        self.root.geometry("2200x1400") 
        try:
            self.root.state('zoomed') # Windows 开启最大化
        except:
            pass
        
        # 获取屏幕分辨率，如果是高分屏，整体字体稍微调大一点
        screen_w = self.root.winfo_screenwidth()
        if screen_w >= 2560:
            ModernStyle.FONT_BODY = ("微软雅黑", 11)
            ModernStyle.FONT_SMALL = ("微软雅黑", 10)
            
        self.root.configure(bg=ModernStyle.BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        ModernStyle.apply_style()
        self.style_treeview()
        
        # Data
        self.global_dataset_path = resource_path("dataset_processed.csv")
        self.user_dataset_path = os.path.join(USER_DATA_DIR, f"{username}_dataset.csv")
        self.df = self.load_data()
        
        self.user_pos = np.array([0.0, 0.0]) 
        self.target_pos = np.array([0.0, 0.0])
        self.rec_mode = tk.StringVar(value="auto")
        self.recommendations_indices = []

        self.create_layout()
        self.init_plot()
        self.update_plot()

    def style_treeview(self):
        style = ttk.Style()
        style.configure("Treeview", 
                        background="white", 
                        foreground="black", 
                        rowheight=26, 
                        fieldbackground="white",
                        font=("微软雅黑", 10))
        style.map('Treeview', background=[('selected', ModernStyle.ACCENT_COLOR)])
        style.configure("Treeview.Heading", font=("微软雅黑", 10, 'bold'))

    def quit_app(self):
        self.root.quit()
        self.root.destroy()
        sys.exit()

    def load_data(self):
        dfs = []
        if os.path.exists(self.global_dataset_path):
            try:
                df1 = pd.read_csv(self.global_dataset_path)
                df1['source'] = 'public'
                # Ensure filename column exists
                if 'filename' not in df1.columns and 'path' in df1.columns:
                     df1['filename'] = df1['path'].apply(os.path.basename)
                dfs.append(self.normalize_df(df1))
            except: pass
        
        if os.path.exists(self.user_dataset_path):
            try:
                df2 = pd.read_csv(self.user_dataset_path)
                df2['source'] = 'private'
                if 'filename' not in df2.columns and 'path' in df2.columns:
                     df2['filename'] = df2['path'].apply(os.path.basename)
                dfs.append(self.normalize_df(df2))
            except: pass

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=['v', 'a', 'filename', 'path', 'v_norm', 'a_norm', 'source'])

    def normalize_df(self, df):
        # 1. 基础归一化到 [-1, 1]
        if 'v_norm' not in df.columns:
            if 'v' in df.columns:
                if df['v'].max() > 1.5: 
                    df['v_norm'] = (df['v'] - 5.5) / 4.5
                    df['a_norm'] = (df['a'] - 5.5) / 4.5
                else: 
                    df['v_norm'] = (df['v'] - 0.5) * 2.0
                    df['a_norm'] = (df['a'] - 0.5) * 2.0
            else:
                 df['v_norm'] = 0
                 df['a_norm'] = 0
        
        # 2. 稀疏化处理 - 提高覆盖率，使音乐散布到边缘
        # 使用 1.6 倍率确保大部分显著音乐能覆盖到 [-1, 1] 边界附近
        df['v_norm'] = (df['v_norm'] * 1.6).clip(-1, 1)
        df['a_norm'] = (df['a_norm'] * 1.6).clip(-1, 1)

        return df
    
    def get_emotion_tag(self, v, a):
        # 按照心理学 V-A 四象限定义
        if v > 0.1 and a > 0.1:
            return "兴奋/愉悦 (Happy)"
        elif v <= -0.1 and a > 0.1:
            return "紧张/愤怒 (Angry)"
        elif v <= -0.1 and a <= -0.1:
            return "忧郁/悲伤 (Sad)"
        elif v > 0.1 and a <= -0.1:
            return "平静/治愈 (Relax)"
        else:
            return "中性/普通 (Neutral)"

    def create_layout(self):
        # Top Bar
        top_bar = ttk.Frame(self.root, style="Main.TFrame", padding=(20, 15))
        top_bar.pack(fill='x')
        ttk.Label(top_bar, text=f"欢迎, {self.username}", font=("微软雅黑", 16, "bold"), background=ModernStyle.BG_COLOR).pack(side='left')
        ttk.Label(top_bar, text="  |  Music Emotion AI System V2.0", font=("微软雅黑", 12), foreground=ModernStyle.SUBTEXT_COLOR, background=ModernStyle.BG_COLOR).pack(side='left')

        # Main Content
        content = ttk.Frame(self.root, style="Main.TFrame")
        content.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # === Left Column (Controls & Upload) ===
        # Width increased to 750 for 2560x1600 resolution and large text
        left_col = ttk.Frame(content, style="Main.TFrame", width=750)
        left_col.pack(side='left', fill='y', padx=(0, 20))
        left_col.pack_propagate(False) # 锁定宽度，确保内容不挤压图标
        
        # [Card 0: Workload / Upload] - MOVED HERE per user request
        c0 = ttk.Frame(left_col, style="Card.TFrame", padding=20)
        c0.pack(fill='x', pady=(0, 15))
        
        ttk.Label(c0, text="▶ 数据工作台 (Workload)", style="Title.TLabel", font=("微软雅黑", 14, "bold")).pack(anchor='w')
        ttk.Label(c0, text="您的私有情绪音乐库管理中心", style="Info.TLabel").pack(anchor='w', pady=(0, 10))
        
        btn_up = tk.Button(c0, text="📂 导入新音乐 / 批量分析特征", 
                           font=("微软雅黑", 10, "bold"),
                           bg="#34495e", # Darker slate for management
                           fg="white",
                           relief="flat",
                           pady=8,
                           cursor="hand2",
                           command=self.open_upload)
        btn_up.pack(fill='x', pady=5)

        btn_manage = tk.Button(c0, text="🔍 查看/编辑现有私有库 (Manage DB)", 
                           font=("微软雅黑", 10),
                           bg="#7f8c8d", 
                           fg="white",
                           relief="flat",
                           pady=5,
                           cursor="hand2",
                           command=self.open_manage)
        btn_manage.pack(fill='x', pady=5)
        
        self.lbl_stats = ttk.Label(c0, text=f"当前库容量: {len(self.df)} 首 (公有+私有)", style="Info.TLabel")
        self.lbl_stats.pack(anchor='w', pady=(5, 0))
        
        # 显示存储路径以便用户查看
        storage_path = os.path.abspath(USER_DATA_DIR)
        ttk.Label(c0, text=f"存储路径: {storage_path}", style="Info.TLabel", font=("微软雅黑", 8)).pack(anchor='w')

        # [Card 1: Analysis] - 尽量压缩高度，给 Step 2 腾位置
        c1 = ttk.Frame(left_col, style="Card.TFrame", padding=16)
        c1.pack(fill='x', pady=(0, 10))
        
        ttk.Label(c1, text="▶ Step 1: 情绪识别 (Emotion Sensing)", style="Subtitle.TLabel").pack(anchor='w')
        
        # 精简说明，减少占用高度
        sensing_desc = ("原理：后端深度模型 (BERT/RoBERTa) 分析语义，\n"
                "将您的文字映射到 Valence-Arousal 情绪平面。")
        ttk.Label(c1, text=sensing_desc, style="Info.TLabel", justify='left', wraplength=650).pack(anchor='w', pady=(4, 8))
        
        ttk.Label(c1, text="请输入您当下的想法或心情描述:", style="Info.TLabel", font=("微软雅黑", 9, "bold")).pack(anchor='w', pady=(5, 5))
        
        self.text_input = tk.Text(c1, height=3, font=("微软雅黑", 11), relief="flat", bg="#f5f7fa", padx=8, pady=6)
        self.text_input.pack(fill='x', pady=(0, 10))
        
        self.btn_analyze = tk.Button(c1, text="✨ 执行 AI 情感计算 (Start Analysis)", 
                         font=("微软雅黑", 10, "bold"),
                                     bg=ModernStyle.ACCENT_COLOR,
                                     fg="white",
                                     relief="flat",
                         pady=6,
                                     cursor="hand2",
                                     command=self.on_analyze_text)
        self.btn_analyze.pack(fill='x')

        # [Card 2: Strategy]
        c2 = ttk.Frame(left_col, style="Card.TFrame", padding=16)
        c2.pack(fill='both', expand=True) # Fill remaining
        
        ttk.Label(c2, text="▶ Step 2: 推荐策略 (Selection Strategy)", style="Subtitle.TLabel").pack(anchor='w')
        
        # 精简版策略说明，控制为两行
        strat_info = ("三种模式：自适应(Auto)、陪伴(Companion)、治愈(Regulate)，\n"
                  "分别用于保持/共鸣/调节当前情绪状态。")
        ttk.Label(c2, text=strat_info, style="Info.TLabel", justify='left', wraplength=650).pack(anchor='w', pady=(4, 8))
        
        self.lbl_strat_desc = ttk.Label(c2, text="请在下方选择一种推荐逻辑：", style="Info.TLabel", font=("微软雅黑", 9, "bold"))
        self.lbl_strat_desc.pack(anchor='w', pady=(0, 10))

        # Radiobuttons：只占用必要高度，确保三种模式全部可见
        r_frame = ttk.Frame(c2, style="Card.TFrame")
        r_frame.pack(fill='x', expand=False)
        self.make_strat_radio(r_frame, "🔮 自适应 (Auto)", "auto", "【自适应】根据当前情绪智能选择目标区域。")
        self.make_strat_radio(r_frame, "🤝 陪伴 (Companion)", "companion", "【陪伴】推荐与当前情绪最相似的音乐。")
        self.make_strat_radio(r_frame, "💊 治愈 (Regulate)", "regulate", "【治愈】强制引导到高愉悦、低唤醒舒适区。")


        # === Right Column (Vis & Recs) ===
        right_col = ttk.Frame(content, style="Card.TFrame", padding=5) # Minimal padding for full look
        right_col.pack(side='right', fill='both', expand=True)
        
        # Split Right into Top (Plot) and Bottom (Table)
        # We need a PanedWindow or just 2 frames. Let's use Frames for rigid layout.
        
        # [Vis Area]
        vis_frame = ttk.Frame(right_col, style="Card.TFrame")
        vis_frame.pack(side='top', fill='both', expand=True, pady=(0, 10))
        
        self.canvas_container = vis_frame
        
        # [Rec Table Area]
        rec_frame = ttk.Frame(right_col, style="Card.TFrame", padding=10)
        rec_frame.pack(side='bottom', fill='x', pady=0)
        
        # Header for Table
        row_h = ttk.Frame(rec_frame, style="Card.TFrame")
        row_h.pack(fill='x', pady=(0, 5))
        ttk.Label(row_h, text="▶ Step 3: 智能推荐结果 (AI Recommendations)", style="Subtitle.TLabel").pack(side='left')
        
        step3_desc = "原理：使用基于欧氏距离与‘显著性权重’的 KNN 算法。表格为您展示了多维度的推荐指标。"
        ttk.Label(rec_frame, text=step3_desc, style="Info.TLabel", font=("微软雅黑", 9)).pack(anchor='w')

        self.lbl_empathy = ttk.Label(rec_frame, text="暂无推荐，请先在左侧输入心情...", font=("微软雅黑", 10, "italic"), foreground=ModernStyle.ACCENT_COLOR)
        self.lbl_empathy.pack(fill='x', pady=(0, 10))
        
        # Treeview for Results (Advanced Table)
        cols_rec = ("title", "va", "type", "score")
        # 控制为中等高度，给播放按钮留足空间
        self.tv_rec = ttk.Treeview(rec_frame, columns=cols_rec, show="headings", height=7)
        
        self.tv_rec.heading("title", text="音乐标题 (Title)")
        self.tv_rec.heading("va", text="情绪坐标 (V, A)")
        self.tv_rec.heading("type", text="情感归属")
        self.tv_rec.heading("score", text="匹配指数")
        
        # Wider columns for 2k monitor
        self.tv_rec.column("title", width=420)
        self.tv_rec.column("va", width=130, anchor='center')
        self.tv_rec.column("type", width=170, anchor='center')
        self.tv_rec.column("score", width=130, anchor='center')
        
        self.tv_rec.pack(fill='x', pady=(0, 10))
        
        # Play Button (Prominent) - 保证始终可见
        self.btn_play = tk.Button(rec_frame, text="▶ 立即播放选中曲目 (Play Selected)", 
                      font=("微软雅黑", 12, "bold"), 
                                  bg=ModernStyle.ACCENT_COLOR, 
                                  fg="white", 
                                  relief="flat", 
                                  activebackground="#2980b9", 
                                  activeforeground="white",
                                  command=self.play_music,
                                  cursor="hand2",
                      pady=14)
        self.btn_play.pack(fill='x')


    def make_strat_radio(self, parent, text, value, desc):
        def on_click():
            self.lbl_strat_desc.config(text=desc)
            if not self.df.empty and np.any(self.user_pos):
                self.update_rec()
                
        rb = ttk.Radiobutton(parent, text=text, variable=self.rec_mode, value=value, command=on_click)
        rb.pack(anchor='w', pady=8)

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def draw_background(self):
        self.ax.clear()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        self.ax.axhline(0, color='#bdc3c7', linestyle='--', linewidth=1)
        self.ax.axvline(0, color='#bdc3c7', linestyle='--', linewidth=1)
        
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        
        self.ax.set_xlabel("Valence (Negative -> Positive)", fontsize=9, color='#7f8c8d')
        self.ax.set_ylabel("Arousal (Calm -> Excited)", fontsize=9, color='#7f8c8d')
        self.ax.set_title("User vs Music in V-A Space", fontsize=12, pad=10, color='#2c3e50')

        # Zones
        self.ax.text(0.9, 0.9, "Happy", ha='right', fontsize=18, color='red', alpha=0.5, fontweight='bold')
        self.ax.text(-0.9, 0.9, "Angry", ha='left', fontsize=18, color='orange', alpha=0.5, fontweight='bold')
        self.ax.text(-0.9, -0.9, "Sad", ha='left', fontsize=18, color='blue', alpha=0.5, fontweight='bold')
        self.ax.text(0.9, -0.9, "Relax", ha='right', fontsize=18, color='green', alpha=0.5, fontweight='bold')

    def update_plot(self):
        self.draw_background()
        
        ux, uy = self.user_pos
        
        # User
        self.ax.scatter(ux, uy, c='#e74c3c', s=250, marker='o', edgecolors='white', linewidth=3, zorder=100, label='User')
        
        if self.recommendations_indices:
            # Target (if not companion)
            if self.rec_mode.get() != "companion":
                tx, ty = self.target_pos
                self.ax.scatter(tx, ty, c='#2ecc71', marker='x', s=180, linewidth=3, zorder=90, label='Goal')
                self.ax.arrow(ux, uy, (tx-ux)*0.8, (ty-uy)*0.8, head_width=0.03, fc='#2ecc71', ec='#2ecc71', alpha=0.4, linestyle=':')
            
            # Helper to draw music
            rows = self.df.iloc[self.recommendations_indices]
            
            # Draw lines
            for _, row in rows.iterrows():
                self.ax.plot([ux, row['v_norm']], [uy, row['a_norm']], color='#3498db', alpha=0.15)
                
            # Draw points
            # Private: Purple, Public: Yellow
            colors = rows['source'].map({'private': '#9b59b6', 'public': '#f1c40f'})
            self.ax.scatter(rows['v_norm'], rows['a_norm'], c=colors, s=120, edgecolors='white', zorder=95)
            
        self.canvas.draw()

    def on_analyze_text(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("提示", "请输入内容")
            return
            
        def task():
            self.btn_analyze.config(state='disabled', text="正在分析语义...")
            if PREDICT_VA_AVAILABLE:
                try:
                    res = predict_va(text)
                    rv, ra = res['valence'], res['arousal']
                    lang = res.get('language')
                    if lang == 'zh':
                        v = (rv - 0.5) * 2
                        a = (ra - 0.5) * 2
                    else:
                        v = (rv - 3.0) / 2.0
                        a = (ra - 3.0) / 2.0
                except:
                    v, a = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
            else:
                time.sleep(1)
                v, a = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
            
            v = np.clip(v, -1, 1)
            a = np.clip(a, -1, 1)

            # Animation
            steps = 12
            start_v, start_a = self.user_pos
            for i in range(steps):
                t = (i+1)/steps
                self.user_pos = np.array([start_v + (v-start_v)*t, start_a + (a-start_a)*t])
                self.root.after(25*i, self.update_plot)
            
            self.root.after(25*steps, lambda: self.finish_analyze(v, a))
            
        threading.Thread(target=task, daemon=True).start()

    def finish_analyze(self, v, a):
        self.user_pos = np.array([v, a])
        self.btn_analyze.config(state='normal', text="✨ AI 情感计算")
        self.update_rec()

    def update_rec(self):
        if self.df.empty: return
        
        uv, ua = self.user_pos
        mode = self.rec_mode.get()
        tv, ta = uv, ua
        empathy_msg = ""
        
        state_desc = self.get_emotion_tag(uv, ua)

        if mode == "auto":
            if uv < -0.2: 
                tv, ta = 0.5, -0.4
                empathy_msg = f"检测到【{state_desc}】。AI 决策：过滤高噪音乐，为您挑选【温暖/治愈】的旋律，请闭眼感受。"
            elif ua > 0.5:
                tv, ta = 0.2, -0.6
                empathy_msg = f"检测到【{state_desc}】。AI 决策：当前心率可能较快，推荐【平稳/深沉】的音乐助您放松。"
            else:
                tv = min(uv + 0.3, 0.9)
                ta = ua
                empathy_msg = f"检测到【{state_desc}】。AI 决策：太棒了，为您推荐一些【轻快/明亮】的歌曲来保持这份好心情。"
                
        elif mode == "regulate":
            tv, ta = 0.8, -0.6
            empathy_msg = "正在执行【心理调节】程序。强力引导至舒适区，让积压的压力随音符消散。"
            
        elif mode == "companion":
            tv, ta = uv, ua
            empathy_msg = f"【深度陪伴】模式。我听到了您的心声，这些歌也许能代表您现在的感受 ({state_desc})。"

        self.target_pos = np.array([tv, ta])
        self.lbl_empathy.config(text=empathy_msg)
        
        # Rec Calculation with Randomness and Significance Priority
        # 1. Calc distance from target
        dist = np.sqrt((self.df['v_norm'] - tv)**2 + (self.df['a_norm'] - ta)**2)
        
        # 2. Prioritize Significance: 远离中心(0,0)的点通常情绪表达更显著
        # 减去该点模长的一部分，使显著的点在排序中更靠前
        magnitude = np.sqrt(self.df['v_norm']**2 + self.df['a_norm']**2)
        self.df['dist'] = dist - (magnitude * 0.18) # 18% 的重要性偏向显著性
        
        # 3. Get bigger pool (e.g., top 15 matches)
        pool_size = min(15, len(self.df))
        candidates = self.df.nsmallest(pool_size, 'dist')
        
        # 3. Random sample 5 from this pool to vary results
        if len(candidates) >= 5:
            selected = candidates.sample(5) # Random sampling
        else:
            selected = candidates
            
        self.recommendations_indices = selected.index.tolist()
        
        # Update Table (Request 5)
        self.tv_rec.delete(*self.tv_rec.get_children())
        for idx in self.recommendations_indices:
            row = self.df.loc[idx]
            
            # Format Title with Source
            src_icon = "🎵" if row['source'] == 'public' else "🔒[私]"
            title = f"{src_icon} {row['filename']}"
            
            # Format VA
            va_str = f"({row['v_norm']:.2f}, {row['a_norm']:.2f})"
            
            # Type
            e_tag = self.get_emotion_tag(row['v_norm'], row['a_norm'])
            
            # Score (inverse of dist, normalized roughly to 0-100)
            score = max(0, int((2.0 - row['dist']) * 50))
            score_bar = "█" * (score // 10)
            
            # Use idx as iid for robust lookup later
            self.tv_rec.insert("", "end", iid=str(idx), values=(title, va_str, e_tag, f"{score}% {score_bar}"))
            
        self.update_plot()

    def play_music(self):
        sel = self.tv_rec.selection()
        if not sel: 
            messagebox.showinfo("提示", "请先在推荐列表中选中一首歌曲")
            return
            
        # Get item iid (which is our df index)
        try:
            row_idx = int(sel[0])
            path = self.df.loc[row_idx, 'path']
            filename = self.df.loc[row_idx, 'filename']
            
            # --- 核心修复：多策略路径寻址 ---
            full_path = None
            
            # 1. 尝试原始路径（可能是绝对或相对）
            if os.path.isabs(path):
                if os.path.exists(path):
                    full_path = path
            else:
                p = os.path.join(os.getcwd(), path)
                if os.path.exists(p):
                    full_path = p

            # 2. 如果原始路径失效，深度扫描项目内的 audiolist 文件夹
            if not full_path:
                audiolist_root = os.path.join(os.getcwd(), "audiolist")
                if os.path.exists(audiolist_root):
                    for root, dirs, files in os.walk(audiolist_root):
                        if filename in files:
                            full_path = os.path.join(root, filename)
                            break
            
            # 3. 如果还是没有，尝试在整个项目目录下找这个文件名
            if not full_path:
                for root, dirs, files in os.walk(os.getcwd()):
                    if filename in files:
                        full_path = os.path.join(root, filename)
                        break

            if full_path and os.path.exists(full_path):
                self.lbl_empathy.config(text=f"▶ 正在播放: {os.path.basename(full_path)}", foreground=ModernStyle.SUCCESS_COLOR)
                try: os.startfile(full_path)
                except: messagebox.showerror("错误", "系统无法打开该文件，可能没有关联的播放器。")
            else:
                messagebox.showerror("文件遗失", f"找不到文件: {filename}\n数据库记录路径: {path}\n请检查音乐是否在项目 audiolist 文件夹内。")
        except Exception as e:
            messagebox.showerror("错误", f"播放逻辑异常: {str(e)}")

    def open_upload(self):
        UploadWindow(self.root, self.on_new_data, self.username)

    def open_manage(self):
        ManageWindow(self.root, self.username, self.refresh_library)

    def refresh_library(self):
        # Triggered after manage window saves
        self.df = self.load_data()
        self.lbl_stats.config(text=f"当前库容量: {len(self.df)} 首 (公有+私有)")
        self.update_plot()
        
    def on_new_data(self, new_rows, username):
        try:
            if not new_rows:
                return False
                
            user_file = os.path.join(USER_DATA_DIR, f"{username}_dataset.csv")
            new_df = pd.DataFrame(new_rows)
            
            # 统一保存字段
            if os.path.exists(user_file):
                try:
                    old_df = pd.read_csv(user_file)
                    # 强行合并去重（以文件路径为索引）
                    final = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=['path'], keep='last')
                except:
                    final = new_df
            else:
                final = new_df
            
            # 建立目录并保存
            if not os.path.exists(USER_DATA_DIR):
                os.makedirs(USER_DATA_DIR)
            final.to_csv(user_file, index=False)
            
            # 同步更新主界面内存中的数据集
            self.df = self.load_data()
            self.lbl_stats.config(text=f"当前库容量: {len(self.df)} 首 (公有+私有)")
            self.update_plot()
            return True
        except Exception as e:
            messagebox.showerror("IO错误", f"无法写入私有数据库: {str(e)}")
            return False

if __name__ == "__main__":
    LoginWindow()

