import os
import datetime

def generate_tree(startpath, output_file):
    # 获取上一级目录路径
    parent_dir = os.path.abspath(os.path.join(startpath, ".."))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Directory Tree generated on {datetime.datetime.now()}\n")
        f.write(f"Root: {parent_dir}\n\n")
        
        for root, dirs, files in os.walk(parent_dir):
            # 从dirs列表中移除.git目录，这样os.walk就不会进入.git目录
            if '.git' in dirs:
                dirs.remove('.git')
                
            # 获取当前目录相对于父目录的相对路径层级
            level = root.replace(parent_dir, '').count(os.sep)
            indent = '│   ' * level
            
            # 只有不是父目录时才打印目录名
            if root != parent_dir:
                f.write(f"{indent}├── {os.path.basename(root)}/\n")
            
            subindent = '│   ' * (level + 1)
            # 过滤掉png文件
            filtered_files = [file for file in files if not file.lower().endswith('.png')]
            for file in sorted(filtered_files):
                f.write(f"{subindent}├── {file}\n")
                
if __name__ == "__main__":
    current_dir = "."  # Current directory
    output_file = "directory_tree.txt"
    generate_tree(current_dir, output_file)
    print(f"Directory tree has been written to {output_file}")