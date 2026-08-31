import os

# ================= 配置区 =================
TARGET_DIR = "."              # 你要扫描的代码库文件夹路径，"." 表示当前目录
OUTPUT_FILE = "llm_prompt.txt" # 输出的文本文件名

# 允许提取内容的文件后缀
ALLOWED_EXTENSIONS = {'.py', '.yaml', '.yml'}

# 生成文件树和遍历时需要忽略的常见无关文件夹
IGNORE_DIRS = {
    '.git', '__pycache__', 'venv', 'env', 
    '.idea', '.vscode', 'node_modules', 'dist', 'build',
    'outputs', 'wandb', 'checkpoints'
}
# ==========================================

def generate_tree(dir_path, prefix=""):
    """递归生成文件树结构的字符串"""
    tree_str = ""
    try:
        # 获取目录下的所有文件和文件夹，并过滤掉忽略的目录
        items = sorted([
            item for item in os.listdir(dir_path) 
            if item not in IGNORE_DIRS
        ])
    except PermissionError:
        return ""

    for i, item in enumerate(items):
        item_path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        
        # 树枝符号
        connector = "└── " if is_last else "├── "
        tree_str += f"{prefix}{connector}{item}\n"
        
        # 如果是文件夹，继续递归深入
        if os.path.isdir(item_path):
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(item_path, prefix=prefix + extension)
            
    return tree_str

def export_codebase_to_prompt(repo_path, output_file):
    repo_path = os.path.abspath(repo_path)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. 注入给大模型的系统提示词 (System Prompt) 和你的问题占位符
        f.write("你是一个高级程序员和代码架构专家。请阅读以下提供的项目代码库上下文，然后回答我的问题。\n\n")
        
        f.write("================================================================\n")
        f.write("【我的问题 / 任务说明】\n")
        f.write("================================================================\n")
        f.write(">>> [请在此处填写你的具体问题。例如：解释一下整个项目的运行逻辑 / 帮我重构一下 main.py / 为什么 yaml 里的配置没生效？] <<<\n\n\n")
        
        # 2. 写入项目文件树结构（给予大模型全局视野）
        f.write("================================================================\n")
        f.write("【项目文件树结构】(仅展示部分相关层级)\n")
        f.write("================================================================\n")
        f.write(f"{os.path.basename(repo_path)}/\n")
        f.write(generate_tree(repo_path))
        f.write("\n\n")
        
        # 3. 遍历写入文件详细内容
        f.write("================================================================\n")
        f.write("【项目文件详细内容】\n")
        f.write("================================================================\n\n")
        
        file_count = 0
        for root, dirs, files in os.walk(repo_path):
            # 过滤掉不需要遍历的文件夹，修改 dirs 即可影响 os.walk 的行为
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    # 获取相对路径，方便大模型阅读
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    # 使用明确的标识符区分文件
                    f.write(f"--- 💡 File: {rel_path} ---\n")
                    
                    # 根据后缀判断 Markdown 的代码语言高亮
                    lang = "python" if ext == ".py" else "yaml"
                    f.write(f"```{lang}\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as code_file:
                            f.write(code_file.read())
                    except Exception as e:
                        f.write(f"# Error reading file: {e}\n")
                    
                    f.write(f"\n```\n\n")
                    file_count += 1
                    
        f.write(f"\n# 提示：共读取了 {file_count} 个代码/配置文件。\n")
        f.write("# [End of Context]\n")
        
    print(f"✅ 成功！共提取了 {file_count} 个文件。")
    print(f"📁 结果已保存至: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    export_codebase_to_prompt(TARGET_DIR, OUTPUT_FILE)