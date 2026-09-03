# Foundation Model Enhancement（基础模型增强）

本目录用于整理针对已有视觉基础模型进行结构改进或能力增强的方法。这里关注的重点是基础模型本身发生了什么变化，例如修改网络结构、插入新的交互模块、调整特征表示，或在原有能力之上加入语言条件、任务条件等新能力，而不是仅仅把基础模型应用到某个下游任务。

适合收录的内容包括：

- ViT、DINO、DINOv2、CLIP 视觉编码器、SAM 等模型的结构改进。
- Attention、token、局部与全局特征交互方式的修改。
- 在已有基础模型中插入新模块，以改善 dense representation、local feature 或 global feature。
- 为基础模型增加语言条件、任务条件或其他可复用能力。
- 在不重新设计整套模型体系的前提下，对基础模型进行增强或扩展。

归档时以论文的核心贡献为准。如果工作主要讨论 Prompt、Adapter、LoRA 或面向具体任务的参数高效微调，更适合归入相应的适配类别；如果只是使用 DINO、CLIP 或 SAM 来解决少样本、开放词汇等问题，则应优先按任务设定归档。只有当主要贡献直接作用于基础模型的结构、表示或通用能力时，才放在本目录。

每个具体方法可以建立独立子目录，例如：

```text
Foundation_Model_Enhancement/
├── SteerViT/
├── DINOv2_Enhancement_Method/
├── CLIP_Dense_Feature_Method/
└── SAM_Architecture_Enhancement/
```

`SteerViT` 是本类别中的一个具体方法，后续同类工作可以继续按方法名称并列添加。
