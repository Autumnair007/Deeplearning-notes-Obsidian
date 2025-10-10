# 导入必要的库
import os
import json
import logging
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from ts.torch_handler.base_handler import BaseHandler
# 注意：需要确保 model.py 文件在 MAR 归档中或 Python 路径下可找到
from model import ResNet18Classifier 
import io # 处理字节流需要导入 io
import base64 # 处理 Base64 编码的图像需要导入 base64

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义 ResNet 处理程序类，继承自 TorchServe 的 BaseHandler
class ResnetHandler(BaseHandler):
    """
    用于图像分类的 ResNet-18 处理程序类。
    """
    def __init__(self):
        # 调用父类的初始化方法
        super(ResnetHandler, self).__init__()
        # 初始化状态标志
        self.initialized = False
        # 模型实例变量
        self.model = None
        # 设备（CPU 或 GPU）
        self.device = None
        # 类别索引到名称的映射
        self.idx_to_class = None
        # 图像预处理转换流程
        self.transform = None

    def initialize(self, context):
        """
        初始化模型和加载类别映射文件。
        """
        # 获取 TorchServe 的系统属性
        properties = context.system_properties
        # 根据系统属性和 CUDA 可用性确定运行设备
        self.device = torch.device("cuda:" + str(properties.get("gpu_id"))
                                   if torch.cuda.is_available() and properties.get("gpu_id") is not None
                                   else "cpu")
        
        # 获取模型文件所在的目录
        model_dir = properties.get("model_dir")
        
        # 1. 导入并实例化模型结构
        self.model = ResNet18Classifier() # 实例化我们定义的包装类
        
        # 2. 加载存储的模型参数 *** (这是关键的修复点) ***
        model_file = os.path.join(model_dir, "resnet18-f37072fd.pth") # 权重文件的完整路径
        if os.path.isfile(model_file):
            # 加载原始（裸露）ResNet-18 的 state_dict
            state_dict = torch.load(model_file, map_location=self.device)
            # 将 state_dict 加载到包装类实例内部的 resnet18 属性中
            # 这是因为我们的权重文件是针对裸 ResNet-18 的，而不是针对 ResNet18Classifier 包装类的
            self.model.resnet18.load_state_dict(state_dict) 
            logger.info(f"成功将权重从 {model_file} 加载到 self.model.resnet18")
        else:
            # 如果权重文件不存在，则抛出错误
            raise RuntimeError(f"模型权重文件丢失: {model_file}")
        
        # 将模型移动到目标设备（CPU 或 GPU）
        self.model.to(self.device)
        # 将模型设置为评估模式（不进行梯度计算和 Dropout）
        self.model.eval()
        
        # 3. 加载类别索引到名称的映射文件 (保持不变)
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            # 打开并加载 JSON 文件
            with open(mapping_file_path) as f:
                self.idx_to_class = json.load(f)
        else:
            # 如果映射文件丢失，记录警告
            logger.warning(f"在 {mapping_file_path} 处缺少 index_to_name.json 文件")
            self.idx_to_class = None
        
        # 4. 设置图像预处理转换流程 (保持不变)
        # 使用 ImageNet 预训练模型的标准转换流程
        self.transform = transforms.Compose([
            transforms.Resize(256),             # 将图像短边缩放到 256
            transforms.CenterCrop(224),         # 从中心裁剪出 224x224 的区域
            transforms.ToTensor(),              # 转换为 PyTorch 张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # 使用 ImageNet 均值进行归一化
                                std=[0.229, 0.224, 0.225]) # 使用 ImageNet 标准差进行归一化
        ])
        
        # 标记初始化完成
        self.initialized = True
        logger.info("ResNet-18 模型初始化成功")

    def preprocess(self, data):
        """
        将原始输入数据转换为模型所需的输入格式。
        """
        # 存储处理后的图像张量列表
        images = []
        # 遍历请求中的每一项数据（通常一个请求包含一个或多个图像）
        for row in data:
            # 从请求数据中获取图像，兼容 'data' 和 'body' 字段
            image = row.get("data") or row.get("body")
            
            # 如果图像是 Base64 编码的字符串
            if isinstance(image, str):
                # 解码 Base64 字符串为字节
                image = base64.b64decode(image) 
            
            # 如果图像是字节数组或字节串
            if isinstance(image, (bytearray, bytes)):
                # 从字节流中打开图像
                image = Image.open(io.BytesIO(image)) 
            # 处理其他可能的意外输入格式
            else:
                 logger.error(f"收到了意外的图像格式: {type(image)}")
                 # 尝试作为文件流打开，但这有风险
                 try:
                     image = Image.open(image) 
                 except Exception as e:
                     logger.error(f"无法处理图像输入: {e}")
                     continue # 跳过这个无法处理的图像

            # 应用预处理转换流程
            image = image.convert("RGB") # 确保图像是 RGB 格式
            image_tensor = self.transform(image)
            images.append(image_tensor)
        
        # 如果没有成功处理任何图像
        if not images:
             raise ValueError("请求数据中未找到有效的图像。")

        # 将图像张量列表堆叠成一个批次 (batch) 并移动到目标设备
        return torch.stack(images).to(self.device)

    def inference(self, x):
        """
        对预处理后的数据运行模型推理。
        """
        # 确保模型在正确的设备上（虽然 initialize 中已做，但这里可以作为安全检查）
        # self.model.to(self.device) 
        # 在不计算梯度的上下文中执行推理
        with torch.no_grad():
            predictions = self.model(x)
        return predictions

    def postprocess(self, inference_output):
        """
        处理模型输出，返回带有标签的预测结果。
        """
        # 对模型输出应用 Softmax 得到概率分布
        probabilities = torch.nn.functional.softmax(inference_output, dim=1)
        # 获取概率最高的 top 5 预测结果及其索引
        # 将结果移动到 CPU 并转换为列表，方便后续处理
        topk_prob, topk_indices = torch.topk(probabilities, 5, dim=1)
        
        results = []
        # 遍历批次中的每个图像的预测结果
        for i in range(topk_indices.size(0)):
            result_single = {} 
            probs = topk_prob[i].cpu().tolist() 
            indices = topk_indices[i].cpu().tolist() 

            # 遍历 top 5 结果
            for j in range(len(indices)): 
                idx = indices[j] 
                prob = probs[j]  
                
                class_name = f"Class_{idx}" # 默认类别名称
                
                # 如果类别映射文件已加载
                if self.idx_to_class:
                    # 检查索引是否存在于映射中
                    if str(idx) in self.idx_to_class:
                        # *** 修改点：从列表中提取第二个元素（人类可读名称）***
                        retrieved_value = self.idx_to_class[str(idx)]
                        if isinstance(retrieved_value, list) and len(retrieved_value) > 1:
                            class_name = retrieved_value[1] # 获取列表的第二个元素
                        else:
                            # 如果值不是预期的列表格式，则使用原始值（可能是简单字符串）
                            class_name = retrieved_value 
                            logger.warning(f"索引 {idx} 在 index_to_name.json 中找到的值不是预期的列表格式:{retrieved_value}")
                    else:
                        # 如果索引不在映射中，记录警告
                        logger.warning(f"索引 {idx} 在 index_to_name.json 中未找到，使用默认值。")
                
                # 将类别名称和概率添加到结果字典中
                result_single[f"class_{j+1}"] = class_name
                result_single[f"probability_{j+1}"] = prob

            # 将单个图像的结果添加到最终列表
            results.append(result_single) 
        
        # 返回结果列表
        return results