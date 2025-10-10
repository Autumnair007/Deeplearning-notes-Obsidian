---
type: "concept-note"
tags: [nlp, text-preprocessing, tokenizer, vocabulary, code-note]
status: "done"
---
学习资料：[8.2. 文本预处理 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/text-preprocessing.html)

[8.3. 语言模型和数据集 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html#id6)

[NLP BERT GPT等模型中 tokenizer 类别说明详解-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1865689)

------
### **一、 什么是文本预处理？**

文本预处理是自然语言处理（NLP）中的一个基础且至关重要的环节。它指的是在将原始文本数据（通常是非结构化的、"脏"的）输入到机器学习或深度学习模型之前，对其进行一系列**清洗、转换和规范化**的操作。

### **二、 为什么需要文本预处理？**

1.  **降低噪声：** 原始文本通常包含很多与核心信息无关的内容，如HTML标签、特殊字符、格式错误、拼写错误等。这些“噪声”会干扰模型的学习。
2.  **格式统一：** 不同来源的文本格式可能千差万别（如大小写、标点符号用法、空格数量）。统一格式有助于模型将相似但形式不同的内容视为一致。
3.  **特征提取：** 模型通常无法直接处理原始字符串。预处理将文本转换为模型能够理解和处理的数值化表示（如数字索引、向量）。
4.  **降低维度/复杂度：** 通过移除低频词、停用词（Stop Words）或使用词干提取/词形还原等技术，可以减少需要模型处理的特征数量，降低模型复杂度和计算成本。
5.  **提升模型性能：** 干净、规范化的数据能让模型更有效地学习文本中的模式和语义，从而提高模型的准确性和泛化能力。

### **三、 文本预处理的常见步骤（通用流程）**

下面是一般文本预处理可能包含的步骤，具体选择哪些步骤以及执行顺序取决于具体的任务和数据：

1.  **数据获取 (Data Acquisition):**
    *   **概念:** 获取用于处理的原始文本数据。来源可以是本地文件、数据库、网页抓取、API接口等。
    *   **通用做法:** 从指定来源读取数据。需要注意文件编码（如UTF-8）、处理读取错误等。对于网络资源，通常会实现下载和本地缓存机制，避免重复下载。
    *   **代码关联:** `TextDownloader` 类负责从URL下载文本，并实现了简单的本地文件缓存逻辑（检查文件是否存在，存在则直接读取）。

2.  **文本清洗 (Text Cleaning):**
    *   **概念:** 移除文本中不需要的字符或格式。
    *   **通用做法:**
        *   移除HTML标签、XML标记。
        *   移除URL、邮箱地址、特殊符号（如@、#、$ 等，除非它们对任务有意义）。
        *   处理或移除表情符号（Emojis）。
        *   移除或替换数字（取决于任务是否需要数字信息）。
        *   **代码关联:** `TextCleaner` 类中的 `clean` 方法使用正则表达式 (`re.sub`) 移除了除字母和基本标点外的字符。它还包含了 `extract_content` 方法，用于提取文本中特定标记之间的部分（这可以看作一种粗粒度的内容过滤）。

3.  **文本规范化 (Text Normalization):**
    *   **概念:** 将文本转换为更一致、标准的形式。
    *   **通用做法:**
        *   **大小写转换:** 通常将所有文本转换为小写，以使 "Apple" 和 "apple" 被视为相同。
        *   **标点符号处理:** 移除所有标点，或者根据需要保留部分标点（如句号、问号可能包含语义信息）。
        *   **处理缩写/俚语:** 将缩写（如 "don't" -> "do not"）或俚语转换为标准形式（这通常需要预定义的词典）。
        *   **标准化空白:** 将多个连续空格、制表符、换行符替换为单个空格，并移除首尾多余空格。
    *   **代码关联:** `TextCleaner` 的 `clean` 方法执行了小写转换 (`.lower()`) 和空白标准化 (`re.sub(r'\s+', ' ', ...).strip()`)。

4.  **分词 (Tokenization):**
    *   **概念:** 将连续的文本流切分成有意义的单元，称为“词元”或“令牌”（Token）。这是NLP中最核心的步骤之一。
    *   **通用做法:**
        *   **词级分词 (Word Tokenization):** 最常见的方式，将文本切分成单词。通常基于空格和标点符号进行分割。对于中文等没有明显空格分隔的语言，需要使用更复杂的算法（如基于词典的最大匹配、基于统计的HMM、CRF或深度学习模型）。
        *   **字符级分词 (Character Tokenization):** 将文本切分成单个字符。优点是词汇表小，不会遇到未登录词（Out-of-Vocabulary, OOV）问题，但序列会变得非常长，且丢失了词的整体语义。
        *   **子词分词 (Subword Tokenization):** 如BPE (Byte Pair Encoding), WordPiece, SentencePiece。介于词级和字符级之间，将词语切分成更小的有意义单元（如 "preprocessing" -> "pre", "process", "ing"）。能有效处理罕见词和复合词，是现代大型语言模型（如BERT, GPT）常用的方法。
    *   **代码关联:** `Tokenizer` 类提供了简单的 `word`（基于空格分割）和 `char`（转换为字符列表）两种模式的分词。

5.  **构建词汇表 (Vocabulary Building):**
    *   **概念:** 创建一个包含数据集中所有（或满足特定条件的）唯一词元的列表，并将每个词元映射到一个唯一的数字索引。
    *   **通用做法:**
        *   统计所有词元的频率。
        *   **频率过滤:** 移除出现次数低于某个阈值（`min_freq`）的低频词。这有助于减小词汇表规模，去除噪声。有时也会移除频率过高的词（停用词）。
        *   **添加特殊词元:** 在词汇表中加入特殊标记，如：
            *   `<unk>` (Unknown): 用于表示在测试集或实际应用中遇到但词汇表中没有的词。
            *   `<pad>` (Padding): 用于将同一批次（Batch）中的序列填充到相同长度。
            *   `<bos>`/`<sos>` (Beginning/Start of Sequence): 标记序列的开始。
            *   `<eos>` (End of Sequence): 标记序列的结束。
        *   创建从词元到索引 (`token_to_idx`) 和从索引到词元 (`idx_to_token`) 的映射。
    *   **代码关联:** `Vocabulary` 类使用 `collections.Counter` 统计词频，根据 `min_freq` 进行过滤，添加了 `<unk>` 词元，并创建了 `idx_to_token` 列表和 `token_to_idx` 字典。

6.  **编码 (Encoding):**
    *   **概念:** 使用构建好的词汇表，将分词后的词元序列转换为对应的数字索引序列。
    *   **通用做法:** 遍历词元序列，查找每个词元在 `token_to_idx` 中的索引。如果词元不在词汇表中，则使用 `<unk>` 对应的索引。
    *   **代码关联:** `Vocabulary` 类中的 `encode` 方法实现了这个功能，对于未找到的词元，默认返回索引0（即 `<unk>` 的索引）。

7.  **（可选）停用词移除 (Stop Word Removal):**
    *   **概念:** 移除文本中非常常见但通常不携带太多实际语义信息的词（如 "the", "a", "is", "in", "和", "的" 等）。
    *   **通用做法:** 维护一个停用词列表，在分词后将出现在列表中的词元移除。对于某些任务（如情感分析），停用词有时也包含信息，因此不一定会移除。现代深度学习模型通常能自己学习哪些词不重要，所以此步骤在深度学习中不太常用。
    *   **代码关联:** 提供的代码中没有显式移除停用词的步骤。

8.  **（可选）词干提取 (Stemming) / 词形还原 (Lemmatization):**
    *   **概念:** 将单词的不同形式（如 "running", "ran", "runs"）归约为它们的基本形式（词干 "run" 或词元 "run"）。
    *   **词干提取:** 通常使用启发式规则切掉单词的后缀，速度快但可能产生非标准词（如 "studies" -> "studi"）。
    *   **词形还原:** 基于词典和词性分析，将单词转换为其字典中的基本形式（词元），结果更准确但计算成本更高（如 "studies" -> "study"）。
    *   **通用做法:** 应用相应的算法库（如 NLTK, spaCy 提供这些功能）处理分词后的词元。
    *   **代码关联:** 提供的代码中没有包含这两个步骤。

### **四、 数据加载与批处理 (Data Loading & Batching)**

在模型训练阶段，通常需要将处理好的长序列数据分割成小批次（Batch）输入模型。

*   **概念:** 定义如何从整个（编码后的）数据集中采样或分割出固定长度的序列，并将它们组织成批次。
*   **通用做法:**
    *   **序列填充 (Padding):** 由于一个批次内的序列长度通常需要一致，较短的序列会用特殊的 `<pad>` 索引填充到与该批次中最长序列相同的长度。
    *   **采样策略:**
        *   **随机采样:** 从整个数据集中随机选择起始点，截取固定长度的序列。有助于打乱数据，增加模型训练的随机性。
        *   **顺序采样/分区:** 按顺序将整个数据集分割成连续（或有重叠）的固定长度序列。这对于需要保持上下文连续性的模型（如RNN）很重要。
*   **代码关联:** `BatchSampler` 类实现了 `random_sampling` 和 `sequential_partitioning` 两种策略，用于从索引序列中创建批次。`sequential_partitioning` 还支持设置重叠（`overlap`）。

### **五、 流程整合 (Pipeline Integration)**

*   **概念:** 将上述所有步骤（或选定的子集）串联起来，形成一个完整的、可重复执行的处理流程。
*   **通用做法:** 设计一个类或一系列函数来封装整个流程，输入原始数据路径或内容，输出模型所需的批处理数据和词汇表。
*   **代码关联:** `TextProcessingPipeline` 类就是这样一个整合器。它的 `run` 方法按顺序调用了下载、提取、清洗、分词、构建词汇表和编码的步骤。`demonstrate_sampling` 方法则展示了如何使用 `BatchSampler` 对处理结果进行批处理。

### **总结:**

文本预处理是NLP中一项复杂但基础的工作，没有万能的“最佳”流程，需要根据具体任务、数据特点和所使用的模型来选择合适的步骤和技术。你提供的代码实现了一个相对完整且模块化的文本预处理流程，涵盖了从数据获取到批次采样的多个关键环节，并考虑了缓存等实用功能，是理解和实践文本预处理的一个很好的示例。理解这些步骤背后的概念和原因比单纯看懂代码更为重要。

------

### 六、长序列数据的读取策略

#### 核心挑战与解决思路
长序列数据（如整本《时间机器》）由于长度超出模型单次处理能力，需进行拆分处理。核心策略是将长序列划分为等长的子序列，通过不同的采样方式生成训练用的小批量数据。关键考量在于如何平衡数据的覆盖性（遍历完整序列）和随机性（防止过拟合）。

---

#### 两种数据采样方法详解

**1. 随机采样（Random Sampling）**
- **核心思想**：每个小批量数据由原始序列中任意位置的子序列构成，相邻批次的子序列无位置关联。
- **实现要点**：
  - 随机偏移起始点：`corpus[random.randint(0, num_steps - 1):]`
  - 计算可划分子序列数：`(len(corpus) - 1) // num_steps`（-1为标签预留）
  - 打乱子序列起始索引：`random.shuffle(initial_indices)`
  - 标签构造：每个样本的标签为下一个时间步的词元序列（Y = X右移1位）

- **代码示例**：
  ```python
  def seq_data_iter_random(corpus, batch_size, num_steps):
      corpus = corpus[random.randint(0, num_steps-1):]
      num_subseqs = (len(corpus)-1) // num_steps
      initial_indices = list(range(0, num_subseqs*num_steps, num_steps))
      random.shuffle(initial_indices)
      
      def data(pos): return corpus[pos:pos+num_steps]
      
      for i in range(0, batch_size*(num_subseqs//batch_size), batch_size):
          indices = initial_indices[i:i+batch_size]
          X = [data(j) for j in indices]
          Y = [data(j+1) for j in indices]  # 标签右移1位
          yield torch.tensor(X), torch.tensor(Y)
  ```
- **输出示例**：
  ```
  X: tensor([[13,14,15,16,17], [28,29,30,31,32]]) 
  Y: tensor([[14,15,16,17,18], [29,30,31,32,33]])
  ```
  *说明*：每个批次包含随机位置的5步子序列，标签为后移1步的序列。

---

**2. 顺序分区（Sequential Partitioning）**
- **核心思想**：保持小批量间的子序列顺序，相邻批次在原始序列中连续。
- **实现要点**：
  - 随机选择初始偏移量：`offset = random.randint(0, num_steps)`
  - 数据重塑为二维：`Xs = Xs.reshape(batch_size, -1)`（行=批量，列=时间步）
  - 按时间步窗口滑动：`X = Xs[:, i:i+num_steps]`

- **代码示例**：
  ```python
  def seq_data_iter_sequential(corpus, batch_size, num_steps):
      offset = random.randint(0, num_steps)
      num_tokens = ((len(corpus)-offset-1)//batch_size)*batch_size
      Xs = torch.tensor(corpus[offset:offset+num_tokens])
      Ys = torch.tensor(corpus[offset+1:offset+1+num_tokens])
      Xs, Ys = Xs.reshape(batch_size,-1), Ys.reshape(batch_size,-1)
      
      for i in range(0, Xs.shape[1]//num_steps * num_steps, num_steps):
          yield Xs[:,i:i+num_steps], Ys[:,i:i+num_steps]
  ```
- **输出示例**：
  ```
  X: tensor([[0,1,2,3,4], [17,18,19,20,21]])
  Y: tensor([[1,2,3,4,5], [18,19,20,21,22]])
  ```
  *说明*：每批的两个样本在原始序列中相距17步，内部子序列保持连续。

---

#### 方法对比与选择
| 特性       | 随机采样                 | 顺序分区               |
| ---------- | ------------------------ | ---------------------- |
| 数据连续性 | ❌ 相邻批次无位置关联     | ✅ 保持原始顺序         |
| 覆盖性     | ✅ 通过随机偏移提高多样性 | ❌ 依赖初始偏移量选择   |
| 内存效率   | 需存储所有子序列索引     | 直接重塑矩阵，内存高效 |
| 适用场景   | 通用语言模型训练         | 需保持上下文连贯的任务 |

---

#### 数据加载器封装
通过`SeqDataLoader`类统一接口，根据参数选择采样方式：
```python
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        self.data_iter_fn = seq_data_iter_random if use_random_iter else seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    return SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens), vocab
```
- **使用示例**：`data_iter, vocab = load_data_time_machine(32, 5, use_random_iter=True)`

---

#### 关键设计思想
1. **标签构造**：语言模型本质是预测下一时间步，故标签始终为输入序列右移一位。
2. **长度对齐**：通过`(len-1)//num_steps`确保特征-标签对齐，避免长度不匹配。
3. **矩阵重塑**：顺序分区中，将一维序列重塑为`[batch_size, -1]`，便于按列滑动窗口。
4. **随机性控制**：随机采样通过打乱索引实现，顺序分区通过初始偏移量引入部分随机性。
