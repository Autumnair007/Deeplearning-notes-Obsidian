这个文件夹里面所有步骤就是我对TorchServe相关信息的全部整理，大部分都是Gemini2.5Pro生成的，所以效率很高。里面有需要实践操作的地方我都自己试过一遍，没有问题的。

师兄如果看部署的话只需要看**TorchServe-ResNet-18操作步骤.md**就行了，其他的都只是操作步骤的补充和我自己整理的资料，方便以后查看的。

与mnist测试的操作步骤相比：

1.我靠AI写了ResNet-18的模型文件（model.py）和resnet18_handler.py两个文件，其中感觉很有用的debug过程也写在了**TorchServe-ResNet-18补充.md**里面。这两个文件到时候肯定都是自己写的，所以先用个样例方便学习使用。

2.我用ngrok测试了其他电脑连接云主机TorchServe能否正常使用，结果是可以的。

3.我探究了Token的使用并且把方法都详细写在**TorchServe-ResNet-18操作步骤.md**里面，使用Token和不使用Token两个方法都写清楚了，并且我测试过没问题。

下面是各个文件的信息：

**ngrok连接的操作步骤.assets和TorchServe-ResNet-18操作步骤.assets文件夹：**里面是对应的markdown文件的图片存储的地方。

**resnet_18文件夹：**官方示例的文件夹下载下来保存在这里的，供参考和使用。

**ngrok连接的操作步骤.md：**测试其他电脑连接云主机上的TorchServe的具体步骤。

**TorchServe-ResNet-18补充.md**：模型部署问题排查与解决报告，感觉有参考意义，还有AI写的编写 TorchServe Handler 程序的框架与细节==（仅供参考，没有实践过）==。

**TorchServe-ResNet-18操作步骤.md**：完整的ResNet-18模型部署的操作步骤，应该非常详细了。

**TorchServe-Token.md**： 官方关于Token的操作的中文版，具体操作步骤我已经融合在了**TorchServe-ResNet-18操作步骤.md**里面，仅供参考。

**TorchServe部署自定义数据集参考操作.md**：AI生成的TorchServe自己部署的完整操作流程。==（仅供参考，没有实践过）==

**model.py**：ResNet-18的模型代码。

**resnet18_handler.py**：ResNet-18的handler代码，负责服务逻辑的核心部分。
