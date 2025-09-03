# app.py (最终版本 - 适用于 index.html 在同级目录)

import os
import json
from flask import Flask, jsonify, send_from_directory

# --- 路径定义 (已更新) ---
# '.' 代表当前目录，Flask将从这里提供 index.html
ROOT_DIR = '.'
INDEX_FILE = 'index.html'

# status.json 的相对路径保持不变
STATUS_FILE = os.path.join('result', 'web_monitor', 'status.json')

# 初始化 Flask 应用
app = Flask(__name__)


@app.route('/')
def index():
    """
    服务主页面 index.html。
    现在它会从项目根目录 (ROOT_DIR) 提供文件。
    """
    return send_from_directory(ROOT_DIR, INDEX_FILE)


@app.route('/api/data')
def get_data():
    """
    提供核心的监控数据 API (此部分逻辑不变)。
    """
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify(
            {"error": f"status.json not found. Expected at '{STATUS_FILE}'. Training may not have started yet."}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to decode status.json. The file might be being updated."}), 500


def check_paths():
    """
    在启动服务器前，检查新结构下的必要文件和目录是否存在。
    """
    print("-" * 60)
    print("🚀 正在启动监控服务器...")

    current_working_dir = os.getcwd()
    print(f"[*] 当前工作目录: {current_working_dir}")

    # 1. 检查 index.html 文件是否存在于当前目录
    full_index_path = os.path.join(current_working_dir, INDEX_FILE)
    if not os.path.isfile(full_index_path):
        print("\n❌ 错误: 'index.html' 文件未找到！")
        print(f"   Flask期望在当前工作目录中找到它: {full_index_path}")
        print(f"   请确保 'index.html' 与 'app.py' 在同一个文件夹下。")
        print("-" * 60)
        return False

    # 2. 检查 status.json 所在的目录是否存在
    status_dir = os.path.dirname(STATUS_FILE)  # -> 'result/web_monitor'
    full_status_dir_path = os.path.join(current_working_dir, status_dir)
    if not os.path.isdir(full_status_dir_path):
        print(f"\n⚠️ 警告: 监控数据目录 '{full_status_dir_path}' 尚未创建。")
        print(f"   这很正常，如果训练还未开始。训练开始后会自动创建此目录。")

    print("[✔] 路径检查完成。")
    return True


if __name__ == '__main__':
    if check_paths():
        print(f"🌍 请在浏览器中打开: http://127.0.0.1:8000")
        print("-" * 60)
        # debug=True 可以在开发时提供有用的错误信息
        # 如果在生产环境或不希望自动重启，可以改为 debug=False
        app.run(host='0.0.0.0', port=8000, debug=True)