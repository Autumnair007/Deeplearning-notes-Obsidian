import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sam_model import SAM, focal_loss, dice_loss, calculate_iou, AdaptiveLossWeights
import random
import json
import os
from PIL import Image
import torch.nn.functional as F
import gc

# 设置matplotlib字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SegmentationDatasetLoader(Dataset):
    """加载之前生成的分割数据集的数据加载器"""

    def __init__(self, dataset_path="./segmentation_dataset", split="train", split_ratio=0.8):
        self.dataset_path = dataset_path
        self.images_dir = os.path.join(dataset_path, "images")
        self.masks_dir = os.path.join(dataset_path, "masks")
        self.annotations_dir = os.path.join(dataset_path, "annotations")

        # 加载数据集信息
        dataset_info_file = os.path.join(dataset_path, "dataset_info.json")
        with open(dataset_info_file, 'r') as f:
            self.dataset_info = json.load(f)

        # 分割数据集
        samples = self.dataset_info["samples"]
        num_train = int(len(samples) * split_ratio)

        if split == "train":
            self.samples = samples[:num_train]
        else:
            self.samples = samples[num_train:]

        print(f"Loading {split} dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # 加载图像和掩码
        image_path = os.path.join(self.images_dir, sample_info["image_file"])
        mask_path = os.path.join(self.masks_dir, sample_info["mask_file"])

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        # 转换为tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0  # (C, H, W)
        mask = torch.from_numpy(mask).float() / 255.0  # (H, W)

        # 生成提示（使用标注信息）
        prompts = self._generate_prompts_from_annotation(sample_info)

        return image, mask, prompts

    def _generate_prompts_from_annotation(self, sample_info):
        """从标注信息生成提示"""
        prompts = {}

        # 修复：改进提示点选择策略
        if sample_info["positive_points"]:
            # 随机选择1-2个正样本点
            num_pos_points = min(len(sample_info["positive_points"]), random.randint(1, 2))
            selected_pos_points = random.sample(sample_info["positive_points"], num_pos_points)

            point_coords = []
            point_labels = []

            for point in selected_pos_points:
                # 归一化坐标到[0, 1]，并添加少量随机扰动
                x = (point[0] + random.uniform(-2, 2)) / 256
                y = (point[1] + random.uniform(-2, 2)) / 256
                x = max(0, min(1, x))  # 确保在[0,1]范围内
                y = max(0, min(1, y))

                point_coords.append([x, y])
                point_labels.append(1)  # 前景

            # 修复：根据掩码质量决定是否添加负样本点
            if sample_info["negative_points"] and random.random() < 0.3:
                neg_point = random.choice(sample_info["negative_points"])
                x = (neg_point[0] + random.uniform(-2, 2)) / 256
                y = (neg_point[1] + random.uniform(-2, 2)) / 256
                x = max(0, min(1, x))
                y = max(0, min(1, y))

                point_coords.append([x, y])
                point_labels.append(0)  # 背景

            prompts["points"] = torch.tensor(point_coords, dtype=torch.float32)
            prompts["point_labels"] = torch.tensor(point_labels, dtype=torch.long)

        # 修复：减少边界框使用频率，避免过度依赖
        if sample_info["bbox"] and random.random() < 0.1:  # 降低到10%
            bbox = sample_info["bbox"]
            # 归一化边界框坐标，并添加少量扰动
            normalized_bbox = [
                max(0, min(1, bbox[0] / 256 + random.uniform(-0.02, 0.02))),
                max(0, min(1, bbox[1] / 256 + random.uniform(-0.02, 0.02))),
                max(0, min(1, bbox[2] / 256 + random.uniform(-0.02, 0.02))),
                max(0, min(1, bbox[3] / 256 + random.uniform(-0.02, 0.02)))
            ]
            prompts["boxes"] = torch.tensor([normalized_bbox], dtype=torch.float32)

        return prompts


def custom_collate_fn(batch):
    """修复：改进的collate函数，更好地处理不同大小的提示"""
    images, masks, prompts_list = zip(*batch)

    # 堆叠图像和掩码
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)

    # 处理提示 - 找到最大的点数量
    max_points = 0
    has_boxes = False

    for prompts in prompts_list:
        if "points" in prompts:
            max_points = max(max_points, prompts["points"].shape[0])
        if "boxes" in prompts:
            has_boxes = True

    # 如果没有点，至少要有1个
    max_points = max(max_points, 1)

    batch_size = len(batch)

    # 初始化批次提示
    batch_prompts = {}

    # 处理点提示
    batch_points = torch.zeros(batch_size, max_points, 2)
    batch_point_labels = torch.zeros(batch_size, max_points, dtype=torch.long)

    for i, prompts in enumerate(prompts_list):
        if "points" in prompts:
            num_points = prompts["points"].shape[0]
            batch_points[i, :num_points] = prompts["points"]
            batch_point_labels[i, :num_points] = prompts["point_labels"]
        else:
            # 修复：根据掩码中心生成更合理的默认点
            mask = masks[i]
            if mask.sum() > 0:
                # 找到掩码重心作为默认点
                y_indices, x_indices = torch.where(mask > 0.5)
                if len(y_indices) > 0:
                    center_y = y_indices.float().mean() / mask.shape[0]
                    center_x = x_indices.float().mean() / mask.shape[1]
                    batch_points[i, 0] = torch.tensor([center_x, center_y])
                else:
                    batch_points[i, 0] = torch.tensor([0.5, 0.5])
            else:
                batch_points[i, 0] = torch.tensor([0.5, 0.5])
            batch_point_labels[i, 0] = 1  # 前景

    batch_prompts["points"] = batch_points
    batch_prompts["point_labels"] = batch_point_labels

    # 处理边界框（如果有的话）
    if has_boxes:
        batch_boxes = torch.zeros(batch_size, 1, 4)
        for i, prompts in enumerate(prompts_list):
            if "boxes" in prompts:
                batch_boxes[i] = prompts["boxes"]
            else:
                # 修复：根据掩码生成更准确的默认边界框
                mask = masks[i]
                if mask.sum() > 0:
                    y_indices, x_indices = torch.where(mask > 0.5)
                    if len(y_indices) > 0:
                        y_min, y_max = y_indices.min().float(), y_indices.max().float()
                        x_min, x_max = x_indices.min().float(), x_indices.max().float()

                        y_min /= mask.shape[0]
                        y_max /= mask.shape[0]
                        x_min /= mask.shape[1]
                        x_max /= mask.shape[1]

                        batch_boxes[i, 0] = torch.tensor([x_min, y_min, x_max, y_max])
                    else:
                        batch_boxes[i, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
                else:
                    batch_boxes[i, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0])

        batch_prompts["boxes"] = batch_boxes

    return images, masks, batch_prompts


def resize_mask_to_match_prediction(mask, pred_size):
    """将掩码调整到与预测相同的尺寸"""
    if len(mask.shape) == 3:  # (B, H, W)
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
    elif len(mask.shape) == 2:  # (H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 调整到预测尺寸
    resized_mask = F.interpolate(mask, size=pred_size, mode='bilinear', align_corners=False)

    if len(resized_mask.shape) == 4 and resized_mask.shape[1] == 1:
        resized_mask = resized_mask.squeeze(1)  # 移除通道维度

    return resized_mask


def train_sam():
    """修复：改进的SAM训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 检查数据集是否存在
    dataset_path = "./segmentation_dataset"
    if not os.path.exists(dataset_path):
        print("Dataset does not exist! Please run create_dataset.py first")
        return None

    # 修复：改进数据加载器配置
    train_dataset = SegmentationDatasetLoader(dataset_path, split="train")
    val_dataset = SegmentationDatasetLoader(dataset_path, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 恢复合理的批次大小
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=2,  # 使用适度的多进程
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 创建模型
    model = SAM(embed_dim=256).to(device)  # 恢复原始嵌入维度

    # 修复：改进的优化器和学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-8)

    # 修改：将T_max改为10以适应新的epoch数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # 修复：使用自适应损失权重
    adaptive_loss = AdaptiveLossWeights().to(device)
    loss_optimizer = optim.Adam(adaptive_loss.parameters(), lr=1e-3)

    # 修改：训练轮数改为10
    num_epochs = 10
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    # 修复：添加梯度累积
    accumulation_steps = 2

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        adaptive_loss.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_idx, (images, masks, prompts) in enumerate(train_loader):
            # 移动到设备
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # 移动提示到设备
            for key in prompts:
                if isinstance(prompts[key], torch.Tensor):
                    prompts[key] = prompts[key].to(device, non_blocking=True)

            # 前向传播
            pred_masks, pred_iou = model(images, prompts)

            # 调整真实掩码尺寸以匹配预测 (64x64)
            resized_masks = resize_mask_to_match_prediction(masks, (64, 64))

            # 修复：改进损失计算
            losses = []
            for i in range(3):
                focal_l = focal_loss(pred_masks[:, i], resized_masks)
                dice_l = dice_loss(pred_masks[:, i], resized_masks)

                # 使用自适应权重
                mask_loss = adaptive_loss(focal_l, dice_l, torch.tensor(0.0, device=device))
                losses.append(mask_loss)

            # 多选择学习 - 使用最小损失
            min_loss, min_idx = torch.min(torch.stack(losses), dim=0)

            # IoU损失 - 需要在原始尺寸上计算IoU
            with torch.no_grad():
                # 将预测上采样到原始尺寸计算IoU
                upsampled_pred = F.interpolate(
                    pred_masks[:, min_idx].unsqueeze(1),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                true_iou = calculate_iou(upsampled_pred.unsqueeze(1), masks.unsqueeze(1))

            iou_loss = F.smooth_l1_loss(pred_iou[:, min_idx], true_iou)  # 使用Smooth L1损失

            total_loss_batch = min_loss + 0.5 * iou_loss

            # 修复：梯度累积
            total_loss_batch = total_loss_batch / accumulation_steps
            total_loss_batch.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(adaptive_loss.parameters(), max_norm=1.0)

                optimizer.step()
                loss_optimizer.step()
                optimizer.zero_grad()
                loss_optimizer.zero_grad()

            total_train_loss += total_loss_batch.item() * accumulation_steps
            num_train_batches += 1

            if batch_idx % 25 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, "
                      f"Training Loss: {total_loss_batch.item() * accumulation_steps:.4f}")

        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        adaptive_loss.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for images, masks, prompts in val_loader:
                # 移动到设备
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                for key in prompts:
                    if isinstance(prompts[key], torch.Tensor):
                        prompts[key] = prompts[key].to(device, non_blocking=True)

                # 前向传播
                pred_masks, pred_iou = model(images, prompts)

                # 调整真实掩码尺寸以匹配预测
                resized_masks = resize_mask_to_match_prediction(masks, (64, 64))

                # 计算验证损失
                losses = []
                for i in range(3):
                    focal_l = focal_loss(pred_masks[:, i], resized_masks)
                    dice_l = dice_loss(pred_masks[:, i], resized_masks)
                    mask_loss = adaptive_loss(focal_l, dice_l, torch.tensor(0.0, device=device))
                    losses.append(mask_loss)

                min_loss, min_idx = torch.min(torch.stack(losses), dim=0)

                # IoU损失
                upsampled_pred = F.interpolate(
                    pred_masks[:, min_idx].unsqueeze(1),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                true_iou = calculate_iou(upsampled_pred.unsqueeze(1), masks.unsqueeze(1))
                iou_loss = F.smooth_l1_loss(pred_iou[:, min_idx], true_iou)

                total_val_loss += (min_loss + 0.5 * iou_loss).item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} completed.")
        print(f"Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'adaptive_loss_state_dict': adaptive_loss.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, "best_sam_model.pth")
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        scheduler.step()

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'adaptive_loss_state_dict': adaptive_loss.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'final_val_loss': avg_val_loss
    }, "final_sam_model.pth")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SAM Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=150)
    plt.show()

    print("Training completed! Model saved.")

    return model


def test_sam_inference(model_path="best_sam_model.pth"):
    """修复：改进的SAM测试函数，解决图像显示问题"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = SAM(embed_dim=256).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建测试数据集
    test_dataset = SegmentationDatasetLoader("./segmentation_dataset", split="val")

    # 测试推理
    num_test_samples = 5
    for i in range(min(num_test_samples, len(test_dataset))):
        image, gt_mask, prompts = test_dataset[i]

        # 移动到设备
        image = image.unsqueeze(0).to(device)
        for key in prompts:
            if isinstance(prompts[key], torch.Tensor):
                prompts[key] = prompts[key].unsqueeze(0).to(device)

        # 设置图像并预测
        model.set_image(image)
        result = model.predict(prompts)

        # 获取基于IoU分数的最佳掩码
        best_mask_idx = torch.argmax(result["iou_scores"][0])
        predicted_mask = result["masks"][0, best_mask_idx].cpu().numpy()

        # 修复：改进掩码后处理，解决模糊问题
        # 应用阈值处理，使掩码更清晰
        predicted_mask_binary = (predicted_mask > 0.5).astype(np.float32)

        # 修复：改进可视化，解决文字截断问题
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 增加图像宽度

        # 输入图像
        axes[0].imshow(image[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Input Image", fontsize=14, pad=20)  # 增加字体大小和padding
        axes[0].axis('off')

        # 真实掩码
        axes[1].imshow(gt_mask.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Mask", fontsize=14, pad=20)
        axes[1].axis('off')

        # 预测掩码 - 使用二值化版本
        axes[2].imshow(predicted_mask_binary, cmap='gray', vmin=0, vmax=1)
        iou_score = result['iou_scores'][0, best_mask_idx].item()
        axes[2].set_title(f"Predicted Mask\n(IoU: {iou_score:.3f})", fontsize=14, pad=20)
        axes[2].axis('off')

        # 修复：调整布局，确保标题完全显示
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 为标题留出更多空间

        # 保存高质量图像
        plt.savefig(f"test_result_{i}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 额外显示：原始预测掩码（用于调试）
        if i == 0:  # 只为第一个样本显示调试信息
            plt.figure(figsize=(15, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(predicted_mask, cmap='gray')
            plt.title(f"Raw Prediction\n(min: {predicted_mask.min():.3f}, max: {predicted_mask.max():.3f})")
            plt.colorbar()
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(predicted_mask_binary, cmap='gray')
            plt.title("Thresholded (>0.5)")
            plt.colorbar()
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.hist(predicted_mask.flatten(), bins=50, alpha=0.7)
            plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
            plt.title("Prediction Value Distribution")
            plt.xlabel("Prediction Value")
            plt.ylabel("Frequency")
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"debug_prediction_{i}.png", dpi=150, bbox_inches='tight')
            plt.show()

        print(f"Test {i + 1}: IoU scores = {result['iou_scores'][0]}")
        print(f"Test {i + 1}: Prediction range = [{predicted_mask.min():.3f}, {predicted_mask.max():.3f}]")


def main():
    """主函数"""
    print("Starting SAM training...")

    # 训练模型
    trained_model = train_sam()

    if trained_model is not None:
        print("\nTesting trained model...")

        # 测试模型
        test_sam_inference()

        print("Training and testing completed!")
    else:
        print("Training failed, please check if dataset exists.")


if __name__ == "__main__":
    main()