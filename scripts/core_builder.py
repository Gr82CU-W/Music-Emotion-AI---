import os
import subprocess
import sys

def build():
    # 基础配置文件
    main_script = "run_app.py"  # 入口文件名更新
    exe_name = "MusicEmotionAI"
    
    # 静态资源列表 (需要打包进 exe 的文件或文件夹)
    # 格式: (源路径, 目标文件夹)
    datas = [
        ("dataset_processed.csv", "."),
        ("models/checkpoint_latest.pth", "models"),           # 对应 models 目录
        ("models/model_music2emo_unified.pth", "models"),
        ("predict_va", "predict_va"),
        # "audiolist" 文件夹因为体积太大, 在 Windows 上会导致打包失败或生成巨型 EXE
        # 这里移除打包，打包完后请手动将 audiolist 复制到 dist/MusicEmotionAI/ 目录下
    ]

    # 构建 pyinstaller 命令
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",   # 改为目录模式 (onedir)，避免 4GB 文件限制
        "--windowed", # 无控制台窗口
        f"--name={exe_name}",
    ]

    # 包含核心模型代码
    cmd.append("--collect-submodules=core") # 包含 core 文件夹中的模块
    cmd.append("--add-data=core;core")      # 以防万一，手动添加 core 相关数据文件

    # 添加静态资源参数
    for src, dst in datas:
        # 注意：在 onedir 模式下，对于 audiolist 建议分发而不是封死在 exe 里
        # 但为了保持用户逻辑，这里依然保持 add-data
        if os.path.exists(src):
            cmd.append(f"--add-data={src}{os.pathsep}{dst}")
        else:
            print(f"Warning: Resource not found: {src}")

    # 给入口文件
    cmd.append(main_script)

    print("Running command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n成功! 可执行文件在 dist/{exe_name}.exe")
    except subprocess.CalledProcessError as e:
        print(f"\n打包失败: {e}")
    except FileNotFoundError:
        print("\n未找到 pyinstaller，请运行: pip install pyinstaller")

if __name__ == "__main__":
    build()
