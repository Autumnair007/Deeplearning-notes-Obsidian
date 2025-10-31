import os
import datetime

def generate_tree(startpath, output_file):
    # 获取上一级目录路径
    parent_dir = os.path.abspath(os.path.join(startpath, ".."))
    
    # 定义需要排除的目录名
    EXCLUDED_DIRS = ['.git', 'images', '99_Assets (资源文件)']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Directory Tree generated on {datetime.datetime.now()}\n")
        f.write(f"Root: {parent_dir}\n\n")
        
        for root, dirs, files in os.walk(parent_dir):
            # 1. 移除需要排除的目录，os.walk将不会进入这些目录
            # 必须在遍历dirs列表之前修改它
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            
            # 获取当前目录相对于父目录的相对路径层级
            level = root.replace(parent_dir, '').count(os.sep)
            indent = '│   ' * level
            
            # 2. 打印当前目录名 (排除根目录本身)
            if root != parent_dir:
                # 检查当前目录是否是images目录（实际上已经被上面的dirs[:]过滤，但为了安全保留原始逻辑）
                # 这里不需要再次检查 'images'，因为 os.walk 已经排除了它。
                # 也不需要检查 '99_Assets (资源文件)'，原因相同。
                f.write(f"{indent}├── {os.path.basename(root)}/\n")
            
            subindent = '│   ' * (level + 1)
            
            # 3. 打印文件，并过滤png和jpg文件
            # 同样，由于images目录已被排除，不需要再次检查 os.path.basename(root) != 'images'
            
            # 过滤掉png和jpg文件
            filtered_files = [file for file in files if not (file.lower().endswith('.png') or file.lower().endswith('.jpg'))]
            for file in sorted(filtered_files):
                f.write(f"{subindent}├── {file}\n")
                
if __name__ == "__main__":
    current_dir = "."    # Current directory
    output_file = "目录树.txt"
    generate_tree(current_dir, output_file)
    print(f"目录树已写入 {output_file}")