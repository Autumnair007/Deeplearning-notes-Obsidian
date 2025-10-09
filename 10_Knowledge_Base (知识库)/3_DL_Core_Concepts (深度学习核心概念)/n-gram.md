---
type: concept-note
tags:
  - nlp
  - language-model
  - statistics
status: done
topic: N-gram 语言模型
core_assumption: 马尔可夫假设 (一个词只依赖前 N-1 个词)
main_issue: 数据稀疏 (Data Sparsity)
solution_classic: 平滑 (Smoothing), 插值 (Interpolation)
---
学习资料：[自然语言处理中N-Gram模型介绍 - 知乎](https://zhuanlan.zhihu.com/p/32829048)

------

### **N-gram 基本概念**
**N-gram** 是一种基于统计的语言模型，用于预测文本中的下一个词（或字符），其核心思想是：**一个词的出现概率仅依赖于它前面的 \( n-1 \) 个词**。  

#### **1. 定义**
- **N-gram** 表示由 **N 个连续词（或字符）** 组成的序列。  
  - **Unigram (1-gram)**：单个词，如 `["the"]`  
  - **Bigram (2-gram)**：两个连续词，如 `["the", "cat"]`  
  - **Trigram (3-gram)**：三个连续词，如 `["the", "cat", "sat"]`  
  - **4-gram, 5-gram...**：依此类推。

#### **2. 计算概率**
N-gram 模型的核心是计算 **条件概率**，即给定前 \( n-1 \) 个词，第 \( n \) 个词出现的概率：
$$
P(w_n | w_{1}, w_{2}, ..., w_{n-1}) \approx P(w_n | w_{n-N+1}, ..., w_{n-1})
$$
**举例**（Bigram 模型）：

- 句子 `"the cat sat on the mat"`  

- Bigrams:  
  `["the", "cat"]`, `["cat", "sat"]`, `["sat", "on"]`, `["on", "the"]`, `["the", "mat"]`  
  
- 计算 `P("sat" | "cat")`：
  $$
  P(\text{"sat"} | \text{"cat"}) = \frac{\text{Count("cat sat")}}{\text{Count("cat")}}
  $$
  如果 `"cat"` 出现 5 次，其中 `"cat sat"` 出现 3 次，则：
  $$
  P(\text{"sat"} | \text{"cat"}) = \frac{3}{5} = 0.6
  $$

#### **3. 优缺点**
✅ **优点**：
- 简单、计算高效（适合小规模数据）。  
- 可解释性强（直接统计词频）。  

❌ **缺点**：
- **数据稀疏问题**（长 N-gram 在训练数据中可能从未出现，概率为 0）。  
- **无法捕捉长距离依赖**（仅依赖前 \( n-1 \) 个词）。  

#### **4. 改进方法**
- **平滑技术**（如 Laplace Smoothing, Kneser-Ney Smoothing）解决零概率问题。  
- **混合模型**（如 Interpolation：结合 Unigram + Bigram + Trigram）。  
- **神经网络语言模型**（如 RNN, Transformer）替代传统 N-gram。  

---

### **N-gram 与深度学习的关系**
在深度学习（如 RNN/LSTM）之前，N-gram 是主流的语言模型。  
- **N-gram**：基于统计，显式计算概率。  
- **神经网络语言模型**：自动学习词的概率分布，能捕捉更长依赖。  

但 N-gram 仍用于：
- 数据预处理（如构建词表）。  
- 快速 baseline 模型（计算资源少时）。  

---

### **示例代码（计算 Bigram 概率）**
```python
from collections import defaultdict

text = "the cat sat on the mat".split()

# 统计 Bigram
bigram_counts = defaultdict(int)
unigram_counts = defaultdict(int)

for i in range(len(text)-1):
    bigram = (text[i], text[i+1])
    bigram_counts[bigram] += 1
    unigram_counts[text[i]] += 1

# 计算 P("sat" | "cat")
prob = bigram_counts[("cat", "sat")] / unigram_counts["cat"]
print(f'P("sat" | "cat") = {prob}')  # 输出 1.0（因为 "cat sat" 只出现一次）
```

---

### **总结**
- N-gram 是 **基于统计的短距离语言模型**，适合小数据场景。  
- 深度学习（如 Transformer）已取代它处理长距离依赖，但 N-gram 仍用于轻量级任务。  
- 改进方法（平滑、插值）可缓解数据稀疏问题。
