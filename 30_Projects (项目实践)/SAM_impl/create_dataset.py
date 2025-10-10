import numpy as np
import os
import json
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt

# 设置matplotlib字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SyntheticSegmentationDataset:
    """合成分割数据集生成器，用于创建几何形状的分割数据集"""

    def __init__(self, dataset_path="./segmentation_dataset", img_size=256):
        self.dataset_path = dataset_path
        self.img_size = img_size

        # 创建数据集文件夹结构
        self.images_dir = os.path.join(dataset_path, "images")
        self.masks_dir = os.path.join(dataset_path, "masks")
        self.annotations_dir = os.path.join(dataset_path, "annotations")

        for dir_path in [self.images_dir, self.masks_dir, self.annotations_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def convert_to_json_serializable(self, obj):
        """将numpy类型转换为可JSON序列化的类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_json_serializable(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj

    def generate_circle_mask(self, img_size):
        """生成随机圆形掩码"""
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # 随机圆形参数
        center_x = random.randint(img_size // 4, 3 * img_size // 4)
        center_y = random.randint(img_size // 4, 3 * img_size // 4)
        radius = random.randint(20, img_size // 6)

        y, x = np.ogrid[:img_size, :img_size]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[circle_mask] = 255

        return mask, (center_x, center_y, radius)

    def generate_rectangle_mask(self, img_size):
        """生成随机矩形掩码"""
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # 随机矩形参数
        x1 = random.randint(10, img_size // 2)
        y1 = random.randint(10, img_size // 2)
        x2 = random.randint(x1 + 30, img_size - 10)
        y2 = random.randint(y1 + 30, img_size - 10)

        mask[y1:y2, x1:x2] = 255

        return mask, (x1, y1, x2, y2)

    def generate_triangle_mask(self, img_size):
        """生成随机三角形掩码"""
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # 生成三个随机顶点
        pt1 = (random.randint(20, img_size - 20), random.randint(20, img_size // 3))
        pt2 = (random.randint(20, img_size // 2), random.randint(2 * img_size // 3, img_size - 20))
        pt3 = (random.randint(img_size // 2, img_size - 20), random.randint(2 * img_size // 3, img_size - 20))

        points = np.array([pt1, pt2, pt3], np.int32)
        cv2.fillPoly(mask, [points], 255)

        return mask, [pt1, pt2, pt3]

    def generate_ellipse_mask(self, img_size):
        """生成随机椭圆掩码"""
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # 椭圆参数
        center = (random.randint(img_size // 4, 3 * img_size // 4),
                  random.randint(img_size // 4, 3 * img_size // 4))
        axes = (random.randint(20, img_size // 6), random.randint(20, img_size // 6))
        angle = random.randint(0, 180)

        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        return mask, (center, axes, angle)

    def generate_compound_mask(self, img_size):
        """生成多个形状组合的掩码"""
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        shape_info = []

        num_shapes = random.randint(2, 4)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'ellipse'])

            if shape_type == 'circle':
                temp_mask, info = self.generate_circle_mask(img_size)
            elif shape_type == 'rectangle':
                temp_mask, info = self.generate_rectangle_mask(img_size)
            else:  # ellipse
                temp_mask, info = self.generate_ellipse_mask(img_size)

            mask = np.maximum(mask, temp_mask)
            shape_info.append({'type': shape_type, 'params': info})

        return mask, shape_info

    def generate_background_image(self, img_size):
        """生成随机背景图像"""
        # 生成随机纹理背景
        bg_type = random.choice(['noise', 'gradient', 'texture'])

        if bg_type == 'noise':
            image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        elif bg_type == 'gradient':
            image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            for i in range(img_size):
                for j in range(img_size):
                    image[i, j] = [int(255 * i / img_size),
                                   int(255 * j / img_size),
                                   int(255 * (i + j) / (2 * img_size))]
        else:  # texture
            image = np.random.rand(img_size, img_size, 3) * 255
            image = image.astype(np.uint8)

        return image

    def apply_shape_to_image(self, image, mask):
        """将形状应用到背景图像上"""
        # 为形状区域生成不同的颜色
        shape_color = [random.randint(0, 255) for _ in range(3)]

        for c in range(3):
            image[:, :, c] = np.where(mask > 0,
                                      shape_color[c],
                                      image[:, :, c])

        return image

    def generate_annotation_points(self, mask):
        """从掩码中生成标注点（正样本和负样本）"""
        positive_points = []
        negative_points = []

        # 生成正样本点（在掩码内部）
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0:
            num_positive = min(3, len(mask_indices[0]))
            selected_idx = np.random.choice(len(mask_indices[0]), num_positive, replace=False)

            for idx in selected_idx:
                x = int(mask_indices[1][idx])
                y = int(mask_indices[0][idx])
                positive_points.append([x, y])

        # 生成负样本点（在掩码外部）
        bg_indices = np.where(mask == 0)
        if len(bg_indices[0]) > 0:
            num_negative = min(2, len(bg_indices[0]))
            selected_idx = np.random.choice(len(bg_indices[0]), num_negative, replace=False)

            for idx in selected_idx:
                x = int(bg_indices[1][idx])
                y = int(bg_indices[0][idx])
                negative_points.append([x, y])

        return positive_points, negative_points

    def generate_bounding_box(self, mask):
        """从掩码生成边界框"""
        indices = np.where(mask > 0)
        if len(indices[0]) == 0:
            return None

        y_min, y_max = np.min(indices[0]), np.max(indices[0])
        x_min, x_max = np.min(indices[1]), np.max(indices[1])

        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def generate_single_sample(self, sample_idx):
        """生成一个完整的数据样本"""
        # 选择形状类型
        shape_type = random.choice(['circle', 'rectangle', 'triangle', 'ellipse', 'compound'])

        # 生成掩码
        if shape_type == 'circle':
            mask, shape_params = self.generate_circle_mask(self.img_size)
        elif shape_type == 'rectangle':
            mask, shape_params = self.generate_rectangle_mask(self.img_size)
        elif shape_type == 'triangle':
            mask, shape_params = self.generate_triangle_mask(self.img_size)
        elif shape_type == 'ellipse':
            mask, shape_params = self.generate_ellipse_mask(self.img_size)
        else:  # compound
            mask, shape_params = self.generate_compound_mask(self.img_size)

        # 生成背景图像
        image = self.generate_background_image(self.img_size)

        # 应用形状到图像
        image = self.apply_shape_to_image(image, mask)

        # 生成标注
        positive_points, negative_points = self.generate_annotation_points(mask)
        bbox = self.generate_bounding_box(mask)

        # 保存文件
        image_filename = f"image_{sample_idx:06d}.png"
        mask_filename = f"mask_{sample_idx:06d}.png"
        annotation_filename = f"annotation_{sample_idx:06d}.json"

        # 保存图像
        Image.fromarray(image).save(os.path.join(self.images_dir, image_filename))

        # 保存掩码
        Image.fromarray(mask).save(os.path.join(self.masks_dir, mask_filename))

        # 保存标注信息
        annotation_info = {
            "image_file": image_filename,
            "mask_file": mask_filename,
            "shape_type": shape_type,
            "shape_params": shape_params,
            "positive_points": positive_points,
            "negative_points": negative_points,
            "bbox": bbox,
            "image_size": [self.img_size, self.img_size]
        }

        # 转换为可JSON序列化的格式
        annotation_info = self.convert_to_json_serializable(annotation_info)

        with open(os.path.join(self.annotations_dir, annotation_filename), 'w') as f:
            json.dump(annotation_info, f, indent=2)

        return annotation_info

    def generate_dataset(self, num_samples=1000):
        """生成完整的数据集"""
        print(f"Starting dataset generation, total {num_samples} samples...")
        print(f"Dataset save path: {self.dataset_path}")

        dataset_info = {
            "dataset_name": "Synthetic Geometric Shapes Segmentation Dataset",
            "num_samples": num_samples,
            "image_size": self.img_size,
            "shape_types": ["circle", "rectangle", "triangle", "ellipse", "compound"],
            "created_date": "2025-01-10",
            "samples": []
        }

        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} samples...")

            try:
                sample_info = self.generate_single_sample(i)
                dataset_info["samples"].append(sample_info)
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        # 保存数据集元信息
        dataset_info = self.convert_to_json_serializable(dataset_info)
        with open(os.path.join(self.dataset_path, "dataset_info.json"), 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"Dataset generation completed!")
        print(f"- Images folder: {self.images_dir}")
        print(f"- Labels folder: {self.masks_dir}")
        print(f"- Annotations folder: {self.annotations_dir}")

        return dataset_info

    def visualize_samples(self, num_samples=5):
        """可视化生成的数据样本"""
        dataset_info_file = os.path.join(self.dataset_path, "dataset_info.json")

        if not os.path.exists(dataset_info_file):
            print("Dataset info file does not exist, please generate dataset first")
            return

        with open(dataset_info_file, 'r') as f:
            dataset_info = json.load(f)

        samples = dataset_info["samples"][:num_samples]

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample_info in enumerate(samples):
            # 加载图像
            image_path = os.path.join(self.images_dir, sample_info["image_file"])
            mask_path = os.path.join(self.masks_dir, sample_info["mask_file"])

            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

            # 显示原图像
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Original Image - {sample_info["shape_type"]}')
            axes[i, 0].axis('off')

            # 显示掩码
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Mask Label')
            axes[i, 1].axis('off')

            # 显示标注点
            axes[i, 2].imshow(image)
            # 绘制正样本点（绿色）
            for point in sample_info["positive_points"]:
                axes[i, 2].plot(point[0], point[1], 'go', markersize=8)
            # 绘制负样本点（红色）
            for point in sample_info["negative_points"]:
                axes[i, 2].plot(point[0], point[1], 'ro', markersize=8)
            # 绘制边界框
            if sample_info["bbox"]:
                bbox = sample_info["bbox"]
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     fill=False, color='blue', linewidth=2)
                axes[i, 2].add_patch(rect)
            axes[i, 2].set_title('Annotation Info')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_path, "sample_visualization.png"), dpi=150)
        plt.show()


def main():
    """主函数：创建数据集"""
    # 设置随机种子以便复现
    random.seed(42)
    np.random.seed(42)

    # 创建数据集生成器
    dataset_generator = SyntheticSegmentationDataset(
        dataset_path="./segmentation_dataset",
        img_size=256
    )

    # 生成数据集
    dataset_info = dataset_generator.generate_dataset(num_samples=500)

    # 可视化一些样本
    print("\nGenerating visualization samples...")
    dataset_generator.visualize_samples(num_samples=5)

    print(f"\nDataset creation completed!")
    print(f"Dataset path: {dataset_generator.dataset_path}")
    print(f"Number of samples: {dataset_info['num_samples']}")
    print(f"Shape types: {dataset_info['shape_types']}")


if __name__ == "__main__":
    main()