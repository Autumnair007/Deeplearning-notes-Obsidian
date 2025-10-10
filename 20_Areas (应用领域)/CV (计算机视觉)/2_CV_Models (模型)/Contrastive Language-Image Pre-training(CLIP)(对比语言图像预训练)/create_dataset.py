import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_sample_dataset(num_samples=2000, output_dir="sample_data"):
    """
    创建简单的图文对数据集
    包含不同颜色、形状、数字的图片和对应描述
    """
    # 使用当前目录作为基础路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_dir)
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    
    data = []
    
    # 定义一些基本的颜色、形状、数字
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black"]
    shapes = ["circle", "square", "triangle", "rectangle"]
    numbers = list(range(10))
    
    print(f"Creating {num_samples} samples...")
    
    for i in range(num_samples):
        # 创建224x224的图片
        img = Image.new('RGB', (224, 224), 'white')
        draw = ImageDraw.Draw(img)
        
        # 随机选择内容类型
        content_type = random.choice(["shape", "number", "color_block"])
        
        if content_type == "shape":
            color = random.choice(colors)
            shape = random.choice(shapes)
            
            # 绘制形状
            x1, y1 = random.randint(50, 100), random.randint(50, 100)
            x2, y2 = x1 + random.randint(50, 100), y1 + random.randint(50, 100)
            
            if shape == "circle":
                draw.ellipse([x1, y1, x2, y2], fill=color)
                text = f"a {color} {shape}"
            elif shape == "square" or shape == "rectangle":
                draw.rectangle([x1, y1, x2, y2], fill=color)
                text = f"a {color} {shape}"
            elif shape == "triangle":
                points = [(x1, y2), ((x1+x2)//2, y1), (x2, y2)]
                draw.polygon(points, fill=color)
                text = f"a {color} {shape}"
                
        elif content_type == "number":
            number = random.choice(numbers)
            color = random.choice(colors)
            
            # 尝试使用系统字体，如果失败则使用默认字体
            try:
                font_size = random.randint(60, 100)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # 计算文本位置
            bbox = draw.textbbox((0, 0), str(number), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (224 - text_width) // 2
            y = (224 - text_height) // 2
            
            draw.text((x, y), str(number), fill=color, font=font)
            text = f"the number {number} in {color}"
            
        elif content_type == "color_block":
            color = random.choice(colors)
            # 绘制大色块
            draw.rectangle([50, 50, 174, 174], fill=color)
            text = f"a {color} colored square"
        
        # 保存图片
        image_path = os.path.join(output_path, "images", f"image_{i:04d}.jpg")
        img.save(image_path)
        
        # 记录数据
        data.append({
            "image_path": image_path,
            "text": text,
            "id": i
        })
        
        if (i + 1) % 100 == 0:
            print(f"Created {i + 1}/{num_samples} samples")
    
    # 保存数据集信息
    with open(os.path.join(output_path, "dataset.json"), "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Dataset created successfully in {output_path}")
    return data

if __name__ == "__main__":
    # 创建2000个样本的数据集
    create_sample_dataset(2000, "sample_data")
    print("数据集创建完成！")
