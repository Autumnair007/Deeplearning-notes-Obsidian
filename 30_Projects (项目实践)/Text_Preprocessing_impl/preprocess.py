# ========================================================================================
# NLP 文本预处理项目
# ========================================================================================
# 描述：一个完整的文本预处理流程，具有本地文件缓存功能
# 特点：
#   - 文本下载和缓存
#   - 文本清洗和标准化
#   - 分词（词级和字符级）
#   - 构建词汇表并进行频率过滤
#   - 在词元和索引之间进行编码和解码
#   - 长序列采样策略（随机和顺序）
# ========================================================================================

# 导入必要的库
import os  # 用于文件和目录操作
import requests  # 用于发送HTTP请求下载文本
import re  # 用于文本清洗中的正则表达式操作
from collections import Counter  # 用于计算词元频率
import random  # 用于数据加载策略中的随机采样


# ========================================================================================
# 类定义
# ========================================================================================

class TextDownloader:
    """处理从URL下载和缓存文本数据。"""
    
    @staticmethod
    def download(url, save_path):
        """从URL下载文本并保存到本地文件。
        
        参数：
            url (str): 下载文本的URL
            save_path (str): 保存下载文本的本地路径
            
        返回：
            str: 下载的文本内容
        """
        if os.path.exists(save_path):  # 检查文件是否已在本地存在
            with open(save_path, 'r', encoding='utf-8') as f:  # 以读模式打开文件，使用UTF-8编码
                print(f"从缓存文件加载: {save_path}")  # 打印状态消息
                return f.read()  # 返回文件内容
        
        # 如果文件不存在，则下载它
        response = requests.get(url)  # 向URL发送HTTP GET请求
        response.raise_for_status()  # 如果请求失败则抛出异常
        text = response.text  # 从响应中获取文本内容
        
        # 将下载的内容保存到本地文件
        with open(save_path, 'w', encoding='utf-8') as f:  # 以写模式打开文件
            f.write(text)  # 将下载的文本写入文件
        
        print(f"已下载并保存到: {os.path.abspath(save_path)}")  # 打印带有绝对路径的确认信息
        return text  # 返回下载的文本


class TextCleaner:
    """处理文本提取和清洗操作。"""
    
    @staticmethod
    def extract_content(text, start_marker, end_marker):
        """提取指定标记之间的文本。
        
        参数：
            text (str): 完整的文本内容
            start_marker (str): 标记所需内容开始的文本
            end_marker (str): 标记所需内容结束的文本
            
        返回：
            str: 标记之间提取的文本
        """
        start_pos = text.find(start_marker)  # 查找开始标记的位置
        end_pos = text.find(end_marker)  # 查找结束标记的位置
        return text[start_pos:end_pos]  # 返回标记之间的切片
    
    @staticmethod
    def clean(text):
        """通过移除特殊字符和标准化空格来清洗文本。
        
        参数：
            text (str): 要清洗的文本
            
        返回：
            str: 清洗后的文本
        """
        # 转换为小写并移除除某些标点符号外的非字母字符
        cleaned = re.sub(r'[^a-zA-Z\s.,!?]', '', text.lower())  # 只保留字母和基本标点
        # 标准化空白：将多个空格替换为单个空格并修剪
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # 将多个空格替换为单个空格
        return cleaned  # 返回清洗后的文本
    
    @staticmethod
    def save_to_file(text, save_path):
        """将文本保存到文件。
        
        参数：
            text (str): 要保存的文本
            save_path (str): 保存文本的文件路径
            
        返回：
            None
        """
        with open(save_path, 'w', encoding='utf-8') as f:  # 以写模式打开文件
            f.write(text)  # 将文本写入文件
        print(f"文本已保存到: {os.path.abspath(save_path)}")  # 打印确认消息


class Tokenizer:
    """处理使用不同策略的文本分词。"""
    
    @staticmethod
    def tokenize(text, mode='word'):
        """根据指定模式将文本转换为词元。
        
        参数：
            text (str): 要分词的文本
            mode (str): 分词模式 - 'word'或'char'
            
        返回：
            list: 词元列表
        """
        if mode == 'word':  # 如果请求词级分词
            return text.split()  # 按空白分割文本得到词级词元
        elif mode == 'char':  # 如果请求字符级分词
            return list(text)  # 将字符串转换为字符列表
        else:  # 如果指定了无效模式
            raise ValueError("模式必须是'word'或'char'")  # 抛出带解释的错误


class Vocabulary:
    """管理词汇表创建、词元到索引的映射以及编码/解码。"""
    
    def __init__(self, tokens, min_freq=1, unk_token="<unk>"):
        """使用频率过滤从词元初始化词汇表。
        
        参数：
            tokens (list): 用于构建词汇表的词元列表
            min_freq (int): 将词元包含在词汇表中所需的最小频率
            unk_token (str): 用于未知词的词元
        """
        self.unk_token = unk_token  # 存储未知词元符号
        
        # 计算词元频率
        counter = Counter(tokens)  # 从词元列表创建计数器对象
        self.token_freq = counter.most_common()  # 获取按频率排序的词元
        
        # 创建索引到词元的映射（词汇表列表）
        self.idx_to_token = [unk_token]  # 第一个条目（索引0）是未知词元
        self.idx_to_token += [token for token, freq in self.token_freq if freq >= min_freq]  # 添加满足最小频率的词元
        
        # 创建词元到索引的映射（查找字典）
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}  # 将每个词元映射到其索引
    
    def __len__(self):
        """返回词汇表大小。
        
        返回：
            int: 词汇表中的词元数量
        """
        return len(self.idx_to_token)  # 返回词元列表的长度
    
    def encode(self, tokens):
        """将词元列表转换为索引列表。
        
        参数：
            tokens (list): 要编码的词元列表
            
        返回：
            list: 索引列表
        """
        return [self.token_to_idx.get(token, 0) for token in tokens]  # 将每个词元映射到其索引（如果未找到则为0）
    
    def decode(self, indices):
        """将索引列表转换回词元列表。
        
        参数：
            indices (list): 要解码的索引列表
            
        返回：
            list: 词元列表
        """
        return [self.idx_to_token[idx] for idx in indices]  # 将每个索引映射回其词元


class BatchSampler:
    """实现从词元序列采样批次的策略。"""
    
    @staticmethod
    def random_sampling(indices, batch_size, seq_length, drop_last=True):
        """从索引中随机采样固定长度的序列。
        
        参数：
            indices (list): 索引的完整序列
            batch_size (int): 每批的序列数量
            seq_length (int): 每个序列的长度
            drop_last (bool): 是否丢弃不完整的最后一批
            
        返回：
            list: 批次列表，每个批次包含batch_size个序列
        """
        valid_idx = len(indices) - seq_length  # 计算最大有效起始索引
        
        # 验证输入
        if valid_idx <= 0:  # 检查序列是否足够长
            raise ValueError(f"序列长度{seq_length}超过输入长度{len(indices)}")  # 如果太短则抛出错误
        
        batches = []  # 初始化空列表以保存批次
        
        # 计算批次数量
        num_batches = valid_idx // batch_size  # 完整批次的整数除法
        if not drop_last and valid_idx % batch_size > 0:  # 处理余数
            num_batches += 1  # 如果不丢弃，则为余数添加一个批次
        
        # 创建批次
        for i in range(num_batches):  # 遍历每个批次
            batch = []  # 初始化当前批次的空列表
            
            # 计算实际批次大小（最后一批可能较小）
            actual_batch_size = batch_size  # 默认批次大小
            if i == num_batches - 1 and not drop_last:  # 如果这是最后一批且我们不丢弃它
                actual_batch_size = min(batch_size, valid_idx - i * batch_size)  # 为余数调整批次大小
            
            # 为此批次生成序列
            for _ in range(actual_batch_size):  # 对于批次中的每个序列
                start_idx = random.randint(0, valid_idx)  # 随机选择起始位置
                sample = indices[start_idx:start_idx + seq_length]  # 提取指定长度的序列
                batch.append(sample)  # 将序列添加到当前批次
            
            batches.append(batch)  # 将完成的批次添加到批次列表
        
        # 打印状态消息
        print(f"随机采样: 创建了{len(batches)}个批次，每个批次有{batch_size}个长度为{seq_length}的样本")
        return batches  # 返回所有批次
    
    @staticmethod
    def sequential_partitioning(indices, batch_size, seq_length, overlap=0, drop_last=True):
        """将索引分区为可选重叠的序列。
        
        参数：
            indices (list): 索引的完整序列
            batch_size (int): 每批的序列数量
            seq_length (int): 每个序列的长度
            overlap (int): 连续序列之间重叠的词元数量
            drop_last (bool): 是否丢弃不完整的最后序列
            
        返回：
            list: 批次列表，每个批次包含batch_size个序列
        """
        # 验证重叠参数
        if overlap >= seq_length:  # 检查重叠是否有效
            raise ValueError(f"重叠({overlap})必须小于序列长度({seq_length})")  # 如果无效则抛出错误
        
        samples = []  # 初始化所有样本的空列表
        batches = []  # 初始化批次的空列表
        
        # 计算连续序列之间的步长
        stride = seq_length - overlap  # 步长是序列长度减去重叠
        
        # 通过滑动窗口生成所有样本
        for i in range(0, len(indices) - seq_length + 1, stride):  # 以stride步长迭代
            sample = indices[i:i + seq_length]  # 提取固定长度的序列
            samples.append(sample)  # 添加到样本列表
        
        # 如果不丢弃最后一个且有剩余词元，则处理剩余词元
        remaining_tokens = len(indices) - (len(samples) * stride - (len(samples) - 1) * overlap)  # 计算剩余词元
        if not drop_last and remaining_tokens > 0:  # 如果保留不完整序列且我们有剩余词元
            last_sample = indices[len(samples) * stride - (len(samples) - 1) * overlap:]  # 获取剩余词元
            
            # 如有必要，填充以达到seq_length
            if len(last_sample) < seq_length:  # 如果是不完整序列
                last_sample = last_sample + [0] * (seq_length - len(last_sample))  # 用零填充
                
            samples.append(last_sample)  # 将填充的序列添加到样本
        
        # 将样本分组为批次
        for i in range(0, len(samples), batch_size):  # 以batch_size步长遍历样本
            if i + batch_size <= len(samples) or not drop_last:  # 如果是完整批次或保留不完整批次
                batch = samples[i:min(i + batch_size, len(samples))]  # 获取样本批次
                
                # 如有必要，填充最后一批
                if len(batch) < batch_size and not drop_last:  # 如果是不完整批次且保留它
                    batch = batch + [batch[-1]] * (batch_size - len(batch))  # 复制最后一个样本进行填充
                    
                batches.append(batch)  # 将批次添加到批次列表
        
        # 打印状态消息
        print(f"顺序分区: 创建了{len(batches)}个批次，步长为{stride}（重叠{overlap}）")
        return batches  # 返回所有批次


# ========================================================================================
# 端到端文本处理的流程类
# ========================================================================================

class TextProcessingPipeline:
    """结合所有步骤的端到端文本处理流程。"""
    
    def __init__(self, url, original_path, processed_path):
        """使用文件路径初始化流程。
        
        参数：
            url (str): 下载文本的URL
            original_path (str): 保存原始文本的路径
            processed_path (str): 保存处理后文本的路径
        """
        self.url = url  # 存储下载URL
        self.original_path = original_path  # 存储原始文本的路径
        self.processed_path = processed_path  # 存储处理后文本的路径
        self.indices = None  # 将索引初始化为None
        self.vocab = None  # 将词汇表初始化为None
    
    def run(self, tokenize_mode='word', min_freq=2, start_marker="CHAPTER I", end_marker="End of Project Gutenberg"):
        """执行完整的处理流程。
        
        参数：
            tokenize_mode (str): 分词模式（'word'或'char'）
            min_freq (int): 词汇表的最小词元频率
            start_marker (str): 标记内容开始的文本
            end_marker (str): 标记内容结束的文本
            
        返回：
            tuple: (索引, 词汇表)
        """
        # 步骤1：下载文本
        raw_text = TextDownloader.download(self.url, self.original_path)  # 下载并缓存文本
        
        # 步骤2：提取内容
        extracted_text = TextCleaner.extract_content(raw_text, start_marker, end_marker)  # 提取相关部分
        
        # 步骤3：清洗文本
        cleaned_text = TextCleaner.clean(extracted_text)  # 清洗和标准化文本
        TextCleaner.save_to_file(cleaned_text, self.processed_path)  # 保存清洗后的文本
        
        # 步骤4：分词
        tokens = Tokenizer.tokenize(cleaned_text, mode=tokenize_mode)  # 将文本转换为词元
        
        # 步骤5：构建词汇表
        self.vocab = Vocabulary(tokens, min_freq=min_freq)  # 创建带频率过滤的词汇表
        
        # 步骤6：将词元编码为索引
        self.indices = self.vocab.encode(tokens)  # 将词元转换为数值索引
        
        # 返回结果
        return self.indices, self.vocab  # 返回索引和词汇表
    
    def demonstrate_sampling(self, batch_size=4, seq_length=20, overlap=5):
        """演示采样策略。
        
        参数：
            batch_size (int): 演示的批次大小
            seq_length (int): 演示的序列长度
            overlap (int): 顺序分区的重叠
            
        返回：
            None
        """
        if self.indices is None or self.vocab is None:  # 检查流程是否已运行
            raise ValueError("必须先运行流程才能演示采样")  # 如果没有则抛出错误
            
        print("\n===== 采样策略演示 =====")  # 打印部分标题
        
        # 随机采样演示
        print("\n1. 随机采样策略:")  # 打印策略名称
        random_batches = BatchSampler.random_sampling(  # 调用随机采样方法
            indices=self.indices,  # 传递流程中的索引
            batch_size=batch_size,  # 传递批次大小
            seq_length=seq_length,  # 传递序列长度
            drop_last=True  # 丢弃不完整的最后一批
        )
        
        # 打印随机采样的示例
        if random_batches:  # 如果我们有批次
            first_sample = random_batches[0][0]  # 获取第一批的第一个样本
            decoded_sample = " ".join(self.vocab.decode(first_sample))  # 解码为文本
            print(f"第一个样本索引: {first_sample}")  # 打印索引
            print(f"解码文本: {decoded_sample}")  # 打印解码文本
        
        # 顺序分区演示
        print("\n2. 顺序分区策略:")  # 打印策略名称
        sequential_batches = BatchSampler.sequential_partitioning(  # 调用顺序分区方法
            indices=self.indices,  # 传递流程中的索引
            batch_size=batch_size,  # 传递批次大小
            seq_length=seq_length,  # 传递序列长度
            overlap=overlap,  # 传递重叠量
            drop_last=False  # 保留不完整的最后一批
        )
        
        # 打印顺序分区的示例
        if sequential_batches and len(sequential_batches[0]) >= 2:  # 如果我们至少有2个样本
            sample1 = sequential_batches[0][0]  # 获取第一个样本
            sample2 = sequential_batches[0][1]  # 获取第二个样本
            decoded_sample1 = " ".join(self.vocab.decode(sample1))  # 解码第一个样本
            decoded_sample2 = " ".join(self.vocab.decode(sample2))  # 解码第二个样本
            
            print(f"第一个样本索引: {sample1}")  # 打印第一个样本索引
            print(f"第二个样本索引: {sample2}")  # 打印第二个样本索引
            print(f"第一个样本文本: {decoded_sample1}")  # 打印第一个样本文本
            print(f"第二个样本文本: {decoded_sample2}")  # 打印第二个样本文本
            
            # 验证重叠
            overlap_correct = sample1[-overlap:] == sample2[:overlap]  # 检查重叠部分是否匹配
            print(f"重叠验证: {overlap_correct}")  # 打印验证结果


# ========================================================================================
# 主执行
# ========================================================================================

if __name__ == "__main__":
    # 配置
    alice_url = "https://www.gutenberg.org/files/11/11-0.txt"  # 爱丽丝梦游仙境文本的URL
    original_file = "alice_original.txt"  # 原始文本文件的路径
    processed_file = "alice_processed.txt"  # 处理后文本文件的路径
    
    # 创建并运行流程
    pipeline = TextProcessingPipeline(  # 创建流程实例
        url=alice_url,  # 传递URL
        original_path=original_file,  # 传递原始文件路径
        processed_path=processed_file  # 传递处理后文件路径
    )
    
    # 执行完整流程
    indices, vocab = pipeline.run(  # 运行流程并获取结果
        tokenize_mode='word',  # 使用词级分词
        min_freq=3,  # 最小词元频率为3
        start_marker="CHAPTER I",  # 内容开始标记
        end_marker="End of Project Gutenberg"  # 内容结束标记
    )
    
    # 打印摘要信息
    print("\n===== 处理摘要 =====")  # 打印部分标题
    print(f"词汇表大小: {len(vocab)}")  # 打印词汇表大小
    print(f"前10个词元: {vocab.idx_to_token[:10]}")  # 打印最频繁的词元
    print(f"总序列长度: {len(indices)}")  # 打印序列长度
    print(f"前20个索引: {indices[:20]}")  # 打印索引序列的开头
    
    # 演示采样策略
    pipeline.demonstrate_sampling(  # 调用采样演示
        batch_size=4,  # 使用批次大小4
        seq_length=20,  # 使用序列长度20
        overlap=5  # 使用5个词元的重叠
    )
    
    print("\n文本预处理项目成功完成!")  # 打印完成消息
