import os
import sys

try:
    import pyperclip
except ImportError:
    print("错误：缺少 'pyperclip' 库。")
    print("请使用 'pip install pyperclip' 命令进行安装。")
    sys.exit(1)


def consolidate_project_to_clipboard(
        project_root: str,
        excluded_dirs: list[str],
        excluded_files: list[str],
        introductory_text: str = ""
) -> None:
    """
    遍历项目目录，将所有符合条件的文件名和内容整合成一个字符串，
    并在内容前添加指定的引导文本，然后复制到系统剪贴板。

    Args:
        project_root (str): 项目的根目录绝对路径。
        excluded_dirs (list[str]): 需要排除的目录名称列表。
        excluded_files (list[str]): 需要排除的文件名称列表。
        introductory_text (str, optional): 添加在所有文件内容之前的介绍性文本。默认为空字符串。
    """
    consolidated_content = []
    file_count = 0

    print(f"开始整合项目: {project_root}")
    print(f"排除目录: {excluded_dirs}")
    print(f"排除文件: {excluded_files}")
    print("-" * 50)

    for root, dirs, files in os.walk(project_root, topdown=True):
        # --- 高效排除目录 ---
        # 通过在遍历过程中直接修改dirs列表，os.walk将不会进入这些目录
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file in files:
            # --- 排除文件 ---
            if file in excluded_files:
                continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, project_root)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # --- 构建专业、规范的分隔符和内容块 ---
                separator = "=" * 20
                file_block = (
                    f"{separator}\n"
                    f"File: {relative_path.replace(os.sep, '/')}\n"  # 统一使用'/'作为路径分隔符
                    f"{separator}\n\n"
                    f"{content}\n\n"
                )
                consolidated_content.append(file_block)
                file_count += 1
                print(f"  [处理中] {relative_path}")

            except UnicodeDecodeError:
                # 如果文件不是UTF-8编码（例如图片、二进制文件），则跳过
                print(f"  [跳  过] {relative_path} (非文本文件或编码错误)")
            except Exception as e:
                print(f"  [错  误] 读取 {relative_path} 时发生错误: {e}")

    if not consolidated_content:
        print("\n未找到任何可整合的文件。")
        return

    # --- 整合最终输出字符串 ---
    # 将引导文本和所有文件内容连接起来
    final_string = introductory_text + "\n\n" + "".join(consolidated_content)

    try:
        pyperclip.copy(final_string)
        print("-" * 20)
        print(f"成功！已整合 {file_count} 个文件。")
        print(f"总字符数: {len(final_string)}")
        print("内容已复制到系统剪贴板。")
    except pyperclip.PyperclipException as e:
        print("-" * 20)
        print(f"错误：无法将内容复制到剪贴板。: {e}")
        print("您可以手动复制以下整合内容：")
        # 为防止终端输出过长内容，可以选择将其写入文件
        output_filename = "consolidated_output.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_string)
        print(f"内容已保存到文件: {output_filename}")


if __name__ == "__main__":
    # --- 配置区 ---

    # 1. 设置前置引导文本
    #    这段文本会出现在所有代码内容之前，可以根据需要进行修改。
    #    使用三重引号'''...'''可以方便地输入多行文本。
    intro_text = "请按照main.py中代码运行逻辑逐文件逐函数逐部分认真、详细、深入地审查项目代码实现（包括README.md文件），找到是否存在错误、潜在的问题、或需要优化的地方，请详细指出，给出专业、科学、准确的修改方案，并返回需要修改文件的完整代码（不必修改的文件不用返回），修改后的文件不必在注释中说明做了哪些修改，仅针对于当前版本进行清晰解释即可（中文）\n\n项目代码如下:\n\n"
    # intro_text = ""

    # 2. 自动获取项目根目录（即本脚本所在的目录）
    #    确保此脚本放置在您的项目根目录下
    project_directory = os.path.dirname(os.path.abspath(__file__))

    # 3. 定义需要排除的目录列表
    #    - .idea: JetBrains IDEs (PyCharm, IntelliJ, etc.)
    #    - __pycache__: Python 编译的字节码缓存
    #    - .git: Git 版本控制目录
    #    - .vscode: Visual Studio Code 配置目录
    #    - venv, .venv: 常见的Python虚拟环境目录
    #    - build, dist, *.egg-info: Python打包生成目录
    #    - web_monitor: 项目中用于监控的web文件夹
    directories_to_exclude = [
        '.idea',
        '__pycache__',
        '.git',
    ]
    # 动态添加所有以 .egg-info 结尾的目录
    for item in os.listdir(project_directory):
        if item.endswith('.egg-info'):
            directories_to_exclude.append(item)

    # 4. 定义需要排除的文件列表
    #    - 本脚本自身
    #    - .DS_Store: macOS 系统文件
    #    - checkpoint.pth.tar, replay_buffer.pkl: 训练产生的二进制文件
    files_to_exclude = [
        os.path.basename(__file__),
        '.DS_Store',
        'checkpoint.pth.tar',
        'replay_buffer.pkl',
        'status.json',  # 训练状态文件
    ]

    # --- 执行 ---
    consolidate_project_to_clipboard(
        project_root=project_directory,
        excluded_dirs=directories_to_exclude,
        excluded_files=files_to_exclude,
        introductory_text=intro_text
    )