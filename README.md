# 口罩佩戴检测

>浙江大学《人工智能与系统》课程作业，口罩佩戴检测。
>
>项目来源于：<https://mo.zju.edu.cn/workspace/5f747601dfcc16df6114a426?type=app&tab=2>（不确定能不能访问）。

项目介绍可以看 `torch_main.ipynb`。

数据集下载：[datasets](https://wwe.lanzous.com/iZ7bYluahpi)。

这里修改了 `torch_py/MobileNetV1.py` 中网络的结构。

训练和推理代码在 `torch_train.ipynb` 和 `train.py` 中，其中 `torch_train.ipynb` 取代价最低的模型，而 `train.py` 会取验证集准确率最高的模型，效率会降低。

