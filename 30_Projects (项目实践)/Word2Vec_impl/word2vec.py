# -*- coding: utf-8 -*-  # 设置文件编码方式
# 导入必要的库
import requests  # 用于下载数据集
import os  # 文件操作
import re  # 正则表达式处��文本
from nltk.corpus import stopwords  # 停用词库
import nltk  # 自然语言处理工具包
from gensim.models import Word2Vec  # Word2Vec模型库

# 下载nltk停用词数据（首次运行需要下载）
nltk.download('stopwords')  # 下载英文停用词语料库

# 1. 数据集下载
# 使用Penn Tree Bank（PTB）数据集（约1MB）
dataset_url = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt"  # 定义数据集URL地址
save_path = "ptb.train.txt"  # 定义本地保存路径

# 如果��件不存在则下载
if not os.path.exists(save_path):  # 检查文件是否已存在
    print("正在下载数据集...")  # 打印下载提示
    response = requests.get(dataset_url)  # 发送GET请求获取数据
    with open(save_path, "wb") as f:  # 以二进制写入模式打开文件
        f.write(response.content)  # 将下载内容写入文件
    print("下载完成！")  # 打印下载完成提示
else:  # 文件已存在的情况
    print("检测到数据集已存在，跳过下载")  # 打印跳过下载的提示

# 2. 数据预处理
print("\n正在预处理数据...")  # 打印预处理开始提示

# 加载英文停用词表
stop_words = set(stopwords.words('english'))  # 获取英文停用词并转为集合

# 新增自定义停用词（根据数据观察添加）
custom_stopwords = {"<unk>", "n"}  # 处理PTB中的特殊标记
stop_words.update(custom_stopwords)  # 将自定义停用词添加到停用词集合中

processed_sentences = []  # 初始化处理后的句子列表

with open(save_path, "r", encoding="utf-8") as f:  # 以UTF-8编码打开文件
    for line in f:  # 逐行读取文件内容
        # 转换为小写并去除首尾空白
        line = line.strip().lower()  # 去除首尾���格并转为小写

        # 去除非字母字符（保留单词构成）
        line = re.sub(r"[^a-zA-Z\s]", "", line)  # 使用正则表达式去除非字母和非空格字符

        # 分割单词并过滤停用词
        words = [  # 使用列表推导式处理单词
            word for word in line.split()  # 分割句子为单词列表
            if word not in stop_words and len(word) > 1  # 同时过滤单字母
        ]

        # 将有效词序列加入列表
        if len(words) > 2:  # 过滤掉过短的句子
            processed_sentences.append(words)  # 将处理后的单词列表添加到结果中

# 展示预处理后的样例
print("\n预处理后的样例（前2个句子）：")  # 打印示例提示
for i in range(2):  # 循环处理前两个句子
    print(f"句子{i + 1}: {' '.join(processed_sentences[i])}")  # 打印句子示例

# 3. 训练Word2Vec模型（使用Skip-gram）
print("\n正在训练模型...")  # 打印训练开始提示

# 配置模型参数
model = Word2Vec(  # 创建Word2Vec模型实例
    vector_size=100,  # 词向量维度（通常100-300）
    window=5,  # 上下文窗口大小（前后各5个词）
    min_count=5,  # 忽略出现次数<5的低频词
    sg=1,  # 1=Skip-gram, 0=CBOW
    workers=4,  # 并行线程数
    epochs=10  # 训练轮次
)

# 构建词汇表
model.build_vocab(processed_sentences)  # 从处理后的句子中构建词汇表

# 开始训练
model.train(  # 训练模型
    processed_sentences,  # 提供训练数据
    total_examples=model.corpus_count,  # 提供语料库大小
    epochs=model.epochs  # 提供训练轮次
)

print(f"\n训练完成！词汇表大小：{len(model.wv.key_to_index)}")  # 打印训练完成信息和词汇量

# 4. 保存模型
model.save("word2vec_model.model")  # 将模型保存到文件
print("\n模型已保存为：word2vec_model.model")  # 打印保存完成提示

# 5. 示例应用
print("\n相似词查询示例：")  # 打印应用示例标题
test_words = ["company", "market", "president", "bank"]  # 定义测试词列表

for word in test_words:  # 遍历每个测试词
    if word in model.wv.key_to_index:  # 检查词是否在词汇表中
        # 查找最相似的5个词
        similar = model.wv.most_similar(word, topn=5)  # 查找最相似的5个词
        print(f"与 '{word}' 最相似的词：")  # 打印相似词标题
        for term, score in similar:  # 遍历相似词及其分数
            print(f"{term}: {score:.3f}")  # 打印词和相似度分数
        print()  # 打印空行分隔
    else:  # 词不在词汇表中的情况
        print(f"'{word}' 不在词汇表中\n")  # 打印词汇表之外的提示

# 计算词语相似度
word_pairs = [  # 定义词对列表
    ("woman", "man"),  # 女人和男人
    ("stock", "market"),  # 股票和市场
    ("china", "japan")  # 中国和日本
]

for w1, w2 in word_pairs:  # 遍历每个词对
    if w1 in model.wv.key_to_index and w2 in model.wv.key_to_index:  # 检查两个词是否都在词汇表中
        similarity = model.wv.similarity(w1, w2)  # 计算两个词的余弦相似度
        print(f"'{w1}'与'{w2}'的余弦相似度：{similarity:.3f}")  # 打印相似度结果
    else:  # 至少一个词不在词汇表中的情况
        print(f"无法计算 '{w1}' 和 '{w2}' 的相似度")  # 打印无法计算的提示

# 查看词向量
print("\n'stock'的词向量示例（前10维）：")  # 打印词向量示例标题
print(model.wv['stock'][:10])  # 打印stock词向量的前10个维度