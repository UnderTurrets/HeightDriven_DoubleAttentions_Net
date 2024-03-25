# 梗概
这是一个使用了dual-attention network模块和height-attention-network模块，使用ResNet50作为骨干网络的图像语义分割项目。
# 数据集
## 下载
1. [Camvid数据集](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
2. [CityScapes数据集](https://www.cityscapes-dataset.com/)

## 目录结构
- Camvid

Camvid数据集的原图和标签在同一目录下，更名为以下形式：

![](assets\屏幕截图 2024-03-25 054437.png)

- CityScapes

gtFine_trainvaltest是标签，leftImg8bit_trainvaltest是原图

![](\assets\屏幕截图 2024-03-25 054528.png)
![](\assets\屏幕截图 2024-03-25 054535.png)

# 快速开始
如果您不想再花时间自己训练，您可以使用我们预训练的模型，它位于链接：https://pan.baidu.com/s/1ME-IBSmYJfx_9GsYARDicg 
提取码：lfhg

## 配置conf.xml
在conf.xml中填入你自己的数据集路径和模型路径，其中，`model/HDANet_?HAM/name`字段不需要修改，`model/HDANet_?HAM/model_file`字段是保存的模型文件的名字。

## 测试数据集加载器
完成conf.xml的配置之后，运行`db/camvid.py`和`db/cityscapes.py`可以检测数据集加载效果。

## 测试数据集语义分割效果
完成conf.xml的配置之后，运行`network/HDAnet.py`可以检测数据集分割效果。

## 对一个图片进行分割测试
完成conf.xml的配置之后，运行`tests/demo.py`可以检测随意一张图片的分割效果。