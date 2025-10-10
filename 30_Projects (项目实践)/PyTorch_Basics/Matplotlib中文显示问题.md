# 无sudo权限解决Matplotlib中文显示问题 - 操作笔记

## 🎯 问题描述
在无`sudo`权限的Linux服务器上，Matplotlib无法找到中文字体（如SimHei），导致中文显示为方框。

## 📋 前置检查

```bash
# 检查用户权限
groups
sudo -l

# 检查当前可用字体
python -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'hei' in f.name.lower() or 'song' in f.name.lower()])"
```

## 🚀 解决方案步骤

### 步骤1：创建用户字体目录
```bash
# 在用户主目录下创建.fonts文件夹
mkdir -p ~/.fonts
```

### 步骤2：下载开源中文字体
```bash
# 进入字体目录
cd ~/.fonts

# 下载文泉驿微米黑字体（推荐，开源免费）
wget https://github.com/anthonyfok/fonts-wqy-microhei/raw/master/wqy-microhei.ttc

# 或者如果wget不可用，手动下载后通过SFTP/scp上传到 ~/.fonts/ 目录
```

### 步骤3：更新字体缓存
```bash
# 方法A：删除Matplotlib缓存并让其自动重建
CACHE_DIR=$(python -c "import matplotlib as mpl; print(mpl.get_cachedir())")
rm -f $CACHE_DIR/fontlist-v*.json

# 方法B：通过导入matplotlib触发缓存重建
python -c "import matplotlib.pyplot as plt"
```

### 步骤4：在Python代码中配置字体

#### 方案A：全局设置（推荐）
```python
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# 设置字体路径
font_path = '/home/你的用户名/.fonts/wqy-microhei.ttc'  # 替换为你的实际路径
chinese_font = FontProperties(fname=font_path)

# 全局配置
plt.rcParams['font.family'] = [chinese_font.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 正常绘图
plt.plot([1, 2, 3], [1, 4, 9])
plt.xlabel('中文X轴')
plt.ylabel('中文Y轴')
plt.title('中文标题')
plt.show()
```

#### 方案B：局部设置（更灵活）
```python
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/home/你的用户名/.fonts/wqy-microhei.ttc'
chinese_font = FontProperties(fname=font_path)

plt.plot([1, 2, 3], [1, 4, 9])
plt.xlabel('中文X轴', fontproperties=chinese_font)
plt.ylabel('中文Y轴', fontproperties=chinese_font)
plt.title('中文标题', fontproperties=chinese_font)
plt.show()
```

## 🔍 验证配置
```python
# 验证字体是否配置成功
import matplotlib.font_manager as fm

# 检查字体是否加载
font_list = [f.name for f in fm.fontManager.ttflist if 'WenQuanYi' in f.name or 'Microhei' in f.name]
print("可用中文字体:", font_list)

# 简单测试绘图
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.text(0.5, 0.5, '中文测试', fontsize=20, ha='center')
plt.title('字体测试')
plt.show()
```

## 💡 备选方案

### 1. 使用系统已有字体
```python
# 尝试使用系统可能已有的其他字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
```

### 2. 联系管理员安装字体
```bash
# 请管理员执行的命令（Ubuntu/Debian）
sudo apt-get install fonts-wqy-microhei

# 请管理员执行的命令（CentOS/RHEL）
sudo yum install wqy-microhei-fonts
```

## 🛠️ 故障排除

### 问题1：字体缓存重建失败
```bash
# 手动清除缓存
rm -rf ~/.cache/matplotlib/
```

### 问题2：字体文件路径错误
```python
# 确认字体文件路径
import os
font_path = '/home/你的用户名/.fonts/wqy-microhei.ttc'
print("字体文件存在:", os.path.exists(font_path))
```

### 问题3：Matplotlib版本问题
```bash
# 检查Matplotlib版本
python -c "import matplotlib; print(matplotlib.__version__)"
```

## 📌 关键要点

1. **无需sudo权限**：所有操作都在用户主目录下完成
2. **使用开源字体**：文泉驿微米黑是很好的选择
3. **更新字体缓存**：确保Matplotlib能识别新字体
4. **两种配置方式**：全局配置或局部配置
5. **路径要正确**：确保字体文件路径准确无误

## ✅ 成功标志
- 中文显示正常，不再出现方框
- 负号正常显示，不出现方块
- 绘图功能一切正常

按照这个笔记操作，您应该能在无sudo权限的情况下完美解决Matplotlib中文显示问题！