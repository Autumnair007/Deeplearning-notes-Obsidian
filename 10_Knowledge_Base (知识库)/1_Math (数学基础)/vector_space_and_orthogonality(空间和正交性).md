#  空间与正交性笔记（DeepSeek生成）

学习资料：[kenjihiranabe/The-Art-of-Linear-Algebra: Graphic notes on Gilbert Strang's "Linear Algebra for Everyone"](https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra)
![](../../99_Assets%20(资源文件)/images/image-20250411113035500.png)
## 一、向量空间（Vector Space）

### 1. **什么是向量空间？**

   - **定义**：向量空间是由一组向量构成的集合，满足加法和数乘的封闭性（即任意向量的线性组合仍在空间内）。
   - **例子**：  
     - 所有二维向量 $( \mathbb{R}^2 = \left\{ \begin{bmatrix} x \\ y \end{bmatrix} \mid x, y \in \mathbb{R} \right\} )$ 是一个向量空间。
     - 平面内所有可能的箭头（向量）就是 $( \mathbb{R}^2 )$。

### 2. **子空间（Subspace）**
   - **定义**：向量空间的子集，本身也是一个向量空间（满足加法和数乘封闭）。
   - **例子**：  
     - 在 $( \mathbb{R}^3 )$ 中，所有形如 $( \begin{bmatrix} x \\ y \\ 0 \end{bmatrix} )$ 的向量构成一个子空间（即 $( xy )$-平面）。
     - **非例子**：所有 $( \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} )$ 不是子空间（因为 $( \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} )$ 不在其中）。

---

## 二、矩阵的四个基本子空间
以矩阵 $( A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix} )$ 为例：

### 1. **列空间（Column Space, $( C(A))$）**
   - **定义**：矩阵列向量的所有线性组合生成的空间。
   - **例子**：  
     - \( A \) 的列向量是 $( \begin{bmatrix} 1 \\ 3 \end{bmatrix} ) 和 ( \begin{bmatrix} 2 \\ 6 \end{bmatrix} )$（第二个列是第一个的2倍）。
     - 列空间 $( C(A) )$ 是 $( \mathbb{R}^2 )$ 中所有形如 $( c \begin{bmatrix} 1 \\ 3 \end{bmatrix} )$ 的向量（即一条穿过原点的直线）。
   - **几何意义**：矩阵 \( A \) 能将任何输入向量 \( x \) 映射到这条直线上。

### 2. **零空间（Null Space, $( N(A) )$）**
   - **定义**：所有满足 \( Ax = 0 \) 的向量 \( x \) 的集合。
   - **例子**：  
     - 解方程 $( \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ \)。
     - 解得 $( x_1 = -2x_2 )$，所以零空间是 $( \left\{ c \begin{bmatrix} -2 \\ 1 \end{bmatrix} \mid c \in \mathbb{R} \right\} )$（另一条直线）。
   - **几何意义**：零空间的向量被 \( A \) “压缩”为零。

### 3. **行空间（Row Space, $( C(A^T) )$）**
   - **定义**：矩阵行向量的所有线性组合生成的空间。
   - **例子**：  
     - \( A \) 的行向量是 \( [1, 2] \) 和 \( [3, 6] \)（第二行是第一行的3倍）。
     - 行空间是 $( \mathbb{R}^2 )$ 中所有形如 $( c \begin{bmatrix} 1 \\ 2 \end{bmatrix} )$ 的向量（**注意这里是行向量的转置**）。

### 4. **左零空间（Left Null Space, $( N(A^T) )$）**
   - **定义**：所有满足 $( yA = 0 )$ 的行向量 \( y \) 的集合（即 $( A^T y^T = 0 )$）。
   - **例子**：  
     - 解方程 $( [y_1, y_2] \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix} = [0, 0] )$。
     - 解得 $( y_1 = -3y_2 )$，所以左零空间是 $( \left\{ c \begin{bmatrix} -3 \\ 1 \end{bmatrix}^T \mid c \in \mathbb{R} \right\} )$。

---

## 三、正交性（Orthogonality）
### 1. **正交的定义**
   - 两个向量 \( u \) 和 \( v \) 正交，当且仅当它们的点积为零：$( u \cdot v = 0 )$。
   - **例子**：  
     - $( \begin{bmatrix} 1 \\ 1 \end{bmatrix} )$ 和 $( \begin{bmatrix} 1 \\ -1 \end{bmatrix} )$ 是正交的，因为 $( 1 \times 1 + 1 \times (-1) = 0 )$。

### 2. **正交补空间（Orthogonal Complement）**
   - **定义**：一个子空间 \( S \) 的正交补空间 $( S^\perp )$ 是所有与 \( S \) 中向量正交的向量的集合。

   - **关键性质**：  
     - $( \mathbb{R}^n = S \oplus S^\perp )$（直和分解）。
     
   - **例子**：  
     
     - 在 $( \mathbb{R}^3 )$ 中，$( xy )$-平面的正交补空间是 \( z \)-轴。
     
     **误区：*"与 $xy$-平面平行的平面是其正交补空间"***
     
     - **错误原因**：  
       平行平面（如 $z=1$ 平面）中的向量 $\begin{bmatrix} a \\ b \\ 1 \end{bmatrix}$ 与 $xy$-平面中的 $\begin{bmatrix} a \\ b \\ 0 \end{bmatrix}$ 点积为 $a^2 + b^2$，除非 $a = b = 0$，否则不为零。  
       - 只有 $z$-轴（即 $a = b = 0$）完全正交。
     - **几何直观**：  
       正交补空间需要与**整个子空间**正交，而非仅与某个特定平面“平行”。

### 3. **四个子空间的正交关系**
   - **列空间 vs 左零空间**：  
     $( C(A) \perp N(A^T) )$（在 $( \mathbb{R}^m )$ 中）。  
     - 例子：$( A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix} )$ 的列空间是 $( \begin{bmatrix} 1 \\ 3 \end{bmatrix} )$ 的方向，左零空间是 $( \begin{bmatrix} -3 \\ 1 \end{bmatrix} )$ 的方向，两者正交。
   - **行空间 vs 零空间**：  
     $( C(A^T) \perp N(A) )$（在 $( \mathbb{R}^n )$ 中）。  
     - 例子：\( A \) 的行空间是 $( \begin{bmatrix} 1 \\ 2 \end{bmatrix} )$ 的方向，零空间是 $( \begin{bmatrix} -2 \\ 1 \end{bmatrix} )$ 的方向，两者正交。

---

## 四、几何图示
### 例子：\( $A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}$ \)
1. **列空间**：$( \mathbb{R}^2 ) 中的直线 ( y = 3x )（方向向量 ( \begin{bmatrix} 1 \\ 3 \end{bmatrix} )$）。  
2. **左零空间**：与之正交的直线 $( y = -\frac{1}{3}x )$（方向向量 $( \begin{bmatrix} -3 \\ 1 \end{bmatrix} )$）。  
3. **行空间**：$( \mathbb{R}^2 ) 中的直线 ( y = 2x )（方向向量 ( \begin{bmatrix} 1 \\ 2 \end{bmatrix} )）$。  
4. **零空间**：$与之正交的直线 ( y = -\frac{1}{2}x )（方向向量 ( \begin{bmatrix} -2 \\ 1 \end{bmatrix} )）$。

---

### 总结
- **空间**：矩阵的列/行空间是它能“到达”的方向，零空间是它“消灭”的方向。
- **正交性**：四个子空间两两正交，揭示了矩阵映射的对称性。
- **应用**：从解方程到机器学习，这些概念无处不在。试着用具体矩阵（如 $( A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} )$）自己画图验证！

------

## 直和分解

---

### 1. **符号定义**
- **$\mathbb{R}^n$**：$n$ 维实向量空间（所有可能的 $n$ 维实向量构成的集合）。
- **$S$**：$\mathbb{R}^n$ 的一个子空间（例如一条直线、一个平面等）。
- **$S^\perp$**：$S$ 的正交补空间（所有与 $S$ 中向量正交的向量的集合）。
- **$\oplus$**：直和运算，表示空间的唯一分解。

---

### 2. **直和分解的含义**
公式 $( \mathbb{R}^n = S \oplus S^\perp )$ 表示：
- **唯一性**：任何向量 $v \in \mathbb{R}^n$ 可以**唯一**表示为 $v = s + t$，其中 $s \in S$，$t \in S^\perp$。
- **正交性**：$S$ 和 $S^\perp$ 中的向量互相垂直（即 $\forall s \in S, t \in S^\perp$，有 $s \cdot t = 0$）。
- **维度关系**：$\dim(S) + \dim(S^\perp) = n$。

---

### 3. **几何例子**
#### 例子 1：$\mathbb{R}^2$ 中的直线
- 设 $S$ 是 $\mathbb{R}^2$ 中通过原点的直线 $y = 2x$（方向向量 $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$）。
- 其正交补空间 $S^\perp$ 是直线 $y = -\frac{1}{2}x$（方向向量 $\begin{bmatrix} -2 \\ 1 \end{bmatrix}$）。
- **分解**：任意向量 $\begin{bmatrix} a \\ b \end{bmatrix} \in \mathbb{R}^2$ 可唯一表示为：
  $\begin{bmatrix} a \\ b \end{bmatrix} = c_1 \begin{bmatrix} 1 \\ 2 \end{bmatrix} + c_2 \begin{bmatrix} -2 \\ 1 \end{bmatrix}, \quad c_1, c_2 \in \mathbb{R}.$
- （通过解线性方程组确定 $c_1, c_2$）

#### 例子 2：$\mathbb{R}^3$ 中的平面
- 设 $S$ 是 $\mathbb{R}^3$ 中的 $xy$-平面（所有形如 $\begin{bmatrix} x \\ y \\ 0 \end{bmatrix}$ 的向量）。
- 其正交补空间 $S^\perp$ 是 $z$-轴（所有形如 $\begin{bmatrix} 0 \\ 0 \\ z \end{bmatrix}$ 的向量）。
- **分解**：任意向量 $\begin{bmatrix} a \\ b \\ c \end{bmatrix}$ 可唯一表示为：
- $
  \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} a \\ b \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ c \end{bmatrix}.
  $

---

### 4. **为什么成立？**
- **正交补的存在性**：通过 Gram-Schmidt 正交化，总能找到 $S$ 的正交基并扩展为 $\mathbb{R}^n$ 的基。
- **唯一性证明**：假设存在两种分解 $v = s_1 + t_1 = s_2 + t_2$，则 $(s_1 - s_2) = (t_2 - t_1)$。由于左边属于 $S$，右边属于 $S^\perp$，且 $S \cap S^\perp = \{0\}$，故 $s_1 = s_2$，$t_1 = t_2$。

---

### 5. **应用场景**
1. **最小二乘法**：将向量 $b$ 分解为 $Ax$（在列空间 $C(A)$）和残差 $b - Ax$（在左零空间 $N(A^T)$）。
2. **PCA（主成分分析）**：数据空间分解为主成分方向（行空间）和噪声方向（零空间）。
3. **信号处理**：信号空间分解为有效信号和噪声（正交补空间）。

---

### 6. **与四个子空间的关系**
对任意矩阵 $A \in \mathbb{R}^{m \times n}$：
- 在 $\mathbb{R}^n$ 中：$\mathbb{R}^n = C(A^T) \oplus N(A)$（行空间 ⊕ 零空间）。
- 在 $\mathbb{R}^m$ 中：$\mathbb{R}^m = C(A) \oplus N(A^T)$（列空间 ⊕ 左零空间）。

---

### 总结
- **直和分解** $\mathbb{R}^n = S \oplus S^\perp$ 表示空间可唯一分解为互相正交的两部分。
- **几何意义**：类似于将空间"投影"到两个垂直的方向上。
- **应用**：从解方程到数据降维，这一概念是线性代数的基石。
