---
type: "concept-note"
tags: [math, linear-algebra, vector-space, fundamental-theorem]
status: "done"
---
学习资料：[kenjihiranabe/The-Art-of-Linear-Algebra: Graphic notes on Gilbert Strang's "Linear Algebra for Everyone"](https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra)
![](../../99_Assets%20(资源文件)/images/image-20250411113035500.png)
***
## 一、向量空间（Vector Space）
### 1. 什么是向量空间？
- **定义**：一个向量空间是一个由“向量”构成的集合 $V$，以及一个标量域 $F$（通常是实数 $\mathbb{R}$），在这个集合上定义了两种运算：向量加法和标量乘法。这些运算必须满足十条公理（包括加法封闭性、数乘封闭性、加法交换律/结合律、存在零向量和逆元等）。简而言之，向量空间是一个对线性组合运算“封闭”的集合。
- **例子**：
  - 所有二维实向量 $\mathbb{R}^2 = \left\{ \begin{bmatrix} x \\ y \end{bmatrix} \mid x, y \in \mathbb{R} \right\}$ 是一个向量空间。
  - 所有次数不超过 $n$ 的多项式集合也是一个向量空间。
### 2. 子空间（Subspace）
- **定义**：向量空间 $V$ 的一个子集 $W$，如果 $W$ 本身也满足向量空间的所有条件，那么 $W$ 就是 $V$ 的一个子空间。
- **检验法则**：要判断 $W$ 是否为子空间，只需检验三点：
  1. **包含零向量**：$0 \in W$。
  2. **对加法封闭**：若 $u, v \in W$，则 $u+v \in W$。
  3. **对数乘封闭**：若 $u \in W$ 且 $c$ 是任意标量，则 $cu \in W$。
- **例子**：
  - 在 $\mathbb{R}^3$ 中，所有形如 $\begin{bmatrix} x \\ y \\ 0 \end{bmatrix}$ 的向量构成一个子空间（即 $xy$-平面）。
- **非例子**：
  - 所有形如 $\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$ 的向量集合不是子空间，因为它不包含零向量 $\begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$。
  - $\mathbb{R}^2$ 中第一象限的向量集合不是子空间，因为它对数乘不封闭（乘以一个负数会跑到第三象限）。
***
## 二、矩阵的四个基本子空间
对于任意 $m \times n$ 矩阵 $A$，它定义了四个与之相关的基本子空间。以矩阵 $A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}$ ($m=2, n=2$) 为例：
### 1. 列空间（Column Space, $C(A)$）
- **定义**：矩阵 $A$ 的所有列向量的线性组合构成的空间。它是 $\mathbb{R}^m$ 的一个子空间。$C(A) = \{ Ax \mid x \in \mathbb{R}^n \}$。
- **例子**：$A$ 的列向量是 $\begin{bmatrix} 1 \\ 3 \end{bmatrix}$ 和 $\begin{bmatrix} 2 \\ 6 \end{bmatrix}$（第二个列是第一个的2倍）。列空间 $C(A)$ 是 $\mathbb{R}^2$ 中所有形如 $c \begin{bmatrix} 1 \\ 3 \end{bmatrix}$ 的向量（即一条穿过原点的直线）。
- **几何意义**：列空间是矩阵 $A$ 作为线性变换所能“到达”的所有可能输出向量的集合。$Ax=b$ 有解的充要条件是 $b \in C(A)$。
- **维度**：$\dim(C(A)) = \text{rank}(A)$。对于例子中的 $A$，秩为1。
### 2. 零空间（Null Space, $N(A)$）
- **定义**：所有满足 $Ax = 0$ 的向量 $x$ 的集合。它是 $\mathbb{R}^n$ 的一个子空间。
- **例子**：解方程 $\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。解得 $x_1 + 2x_2 = 0$，所以零空间是 $\left\{ c \begin{bmatrix} -2 \\ 1 \end{bmatrix} \mid c \in \mathbb{R} \right\}$（另一条直线）。
- **几何意义**：零空间的向量是所有被矩阵 $A$ “压缩”或“映射”到原点的输入向量。
- **维度**：$\dim(N(A)) = n - \text{rank}(A)$。这被称为**秩-零度定理**。对于例子中的 $A$，$2 - 1 = 1$。
### 3. 行空间（Row Space, $C(A^T)$）
- **定义**：矩阵 $A$ 的所有行向量的线性组合构成的空间（等价于 $A^T$ 的列空间）。它是 $\mathbb{R}^n$ 的一个子空间。
- **例子**：$A$ 的行向量是 $[1, 2]$ 和 $[3, 6]$（第二行是第一行的3倍）。行空间是 $\mathbb{R}^2$ 中所有形如 $c \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ 的向量（**注意这里是行向量的转置**）。
- **维度**：一个非常重要的性质是 $\dim(C(A^T)) = \dim(C(A)) = \text{rank}(A)$。对于例子中的 $A$，维度为1。
### 4. 左零空间（Left Null Space, $N(A^T)$）
- **定义**：所有满足 $A^T y = 0$ 的向量 $y$ 的集合（等价于满足 $y^T A = 0^T$ 的行向量 $y^T$ 的转置）。它是 $\mathbb{R}^m$ 的一个子空间。
- **例子**：解方程 $\begin{bmatrix} 1 & 3 \\ 2 & 6 \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。解得 $y_1 + 3y_2 = 0$，所以左零空间是 $\left\{ c \begin{bmatrix} -3 \\ 1 \end{bmatrix} \mid c \in \mathbb{R} \right\}$。
- **维度**：$\dim(N(A^T)) = m - \text{rank}(A)$。对于例子中的 $A$，$2 - 1 = 1$。
***
## 三、正交性与线性代数基本定理
### 1. 正交的定义
- 两个向量 $u$ 和 $v$ 正交，当且仅当它们的点积为零：$u \cdot v = u^T v = 0$。
- **例子**：$\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 和 $\begin{bmatrix} 1 \\ -1 \end{bmatrix}$ 是正交的，因为 $1 \times 1 + 1 \times (-1) = 0$。
### 2. 正交补空间（Orthogonal Complement）
- **定义**：一个子空间 $S$ 的正交补空间 $S^\perp$ 是所有与 $S$ 中**每一个**向量都正交的向量的集合。
- **关键性质**：$S$ 和 $S^\perp$ 都是子空间，并且它们的交集只有零向量 $S \cap S^\perp = \{0\}$。
- **例子**：在 $\mathbb{R}^3$ 中，$xy$-平面的正交补空间是 $z$-轴。
### 3. 四个子空间的正交关系（线性代数基本定理，第二部分）
这四个子空间之间存在着深刻而优美的正交关系：
- **行空间与零空间正交**：$C(A^T) \perp N(A)$。它们共同构成了输入空间 $\mathbb{R}^n$。
  - **证明**：若 $v \in C(A^T)$，$x \in N(A)$，则 $v = A^T z$ 且 $Ax=0$。那么 $v^T x = (A^T z)^T x = z^T A x = z^T (Ax) = z^T 0 = 0$。
- **列空间与左零空间正交**：$C(A) \perp N(A^T)$。它们共同构成了输出空间 $\mathbb{R}^m$。
  - **证明**：同理，这等价于 $C((A^T)^T) \perp N(A^T)$。
***
## 四、直和分解与投影
### 1. 直和分解的含义
公式 $\mathbb{R}^n = S \oplus S^\perp$ 表示：任何向量 $v \in \mathbb{R}^n$ 都可以被**唯一地**分解为一个在 $S$ 中的分量和一个在 $S^\perp$ 中的分量。
$$
v = v_S + v_{S^\perp} \quad \text{其中 } v_S \in S, \ v_{S^\perp} \in S^\perp
$$
- $v_S$ 称为 $v$ 在子空间 $S$ 上的**正交投影**。
- **维度关系**：$\dim(S) + \dim(S^\perp) = n$。
### 2. 几何例子
- **$\mathbb{R}^2$ 中的直线**：设 $S$ 是直线 $y=2x$。其正交补 $S^\perp$ 是直线 $y = -1/2 x$。任何二维向量都可以唯一地表示为这两个方向上的向量之和。
- **$\mathbb{R}^3$ 中的平面**：设 $S$ 是 $xy$-平面。其正交补 $S^\perp$ 是 $z$-轴。任何三维向量 $\begin{bmatrix} a \\ b \\ c \end{bmatrix}$ 可唯一分解为 $\begin{bmatrix} a \\ b \\ 0 \end{bmatrix} \in S$ 和 $\begin{bmatrix} 0 \\ 0 \\ c \end{bmatrix} \in S^\perp$。
### 3. 与四个子空间的关系（线性代数基本定理，第一和第二部分）
对任意矩阵 $A \in \mathbb{R}^{m \times n}$：
- **输入空间分解**：$\mathbb{R}^n = C(A^T) \oplus N(A)$ (行空间 $\oplus$ 零空间)。
- **输出空间分解**：$\mathbb{R}^m = C(A) \oplus N(A^T)$ (列空间 $\oplus$ 左零空间)。
***
## 五、线性映射的完整图景
结合以上所有概念，我们可以描绘出矩阵 $A$ 作为线性映射 $T(x) = Ax$ 的完整行为：
1.  矩阵 $A$ 将其**行空间** $C(A^T)$ 中的向量**一一映射**到其**列空间** $C(A)$ 中的向量。这是一个可逆的变换。
2.  矩阵 $A$ 将其**零空间** $N(A)$ 中的所有向量都**映射到零向量** $0 \in \mathbb{R}^m$。
3.  任何输入向量 $x \in \mathbb{R}^n$ 可以被分解为 $x = x_{row} + x_{null}$，其中 $x_{row} \in C(A^T)$，$x_{null} \in N(A)$。
4.  经过映射后，$Ax = A(x_{row} + x_{null}) = Ax_{row} + Ax_{null} = Ax_{row} + 0 = Ax_{row}$。这意味着，向量 $x$ 的零空间分量 $x_{null}$ 在映射中被“消灭”了，只有其行空间分量 $x_{row}$ 对最终的输出有贡献。
### 总结
- **空间**：矩阵的行空间是其有效输入的“舞台”，列空间是其有效输出的“成像”，零空间是被“抹去”的信息，左零空间是输出空间中无法被“触及”的区域。
- **正交性**：揭示了输入和输出空间内在的几何结构，有效信息与被丢弃的信息是相互垂直的。
- **应用**：这个框架是最小二乘法、主成分分析（PCA）、奇异值分解（SVD）等众多高级应用的理论基石。例如，在最小二乘法中，我们正是将向量 $b$ 投影到列空间 $C(A)$ 上来找到最佳近似解。