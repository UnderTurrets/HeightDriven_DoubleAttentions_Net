'''
Created by Han Xu
email:736946693@qq.com
'''
import xml.etree.ElementTree as ET


def element_to_dict(element):
    """将XML元素递归转换为字典"""
    result = {}

    for child in element:
        child_dict = element_to_dict(child)

        if child.tag in result:
            # 如果标签已经存在于字典中，将其转换为列表
            if type(result[child.tag]) is list:
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = [result[child.tag], child_dict]
        else:
            result[child.tag] = child_dict

    # 使用strip去掉两边的空白字符，如果有文本内容
    text = element.text.strip() if element.text else None

    # 如果有文本内容，且没有子元素，直接存储文本内容
    if text and not result:
        return text

    # 如果有文本内容，添加到字典中
    if text:
        result[element.tag] = text

    return result


# 读取配置信息
import os

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 切换到该目录
os.chdir(current_directory)
# 读取XML文件
tree = ET.parse("conf.xml")
root = tree.getroot()

# 将XML转换为嵌套字典
xml_dict = element_to_dict(root)

# 需要的变量
camvid_path = xml_dict["dataset"]["camvid_path"]
model_dict = xml_dict["model"]
if __name__ == "__main__":
    # 打印嵌套字典
    import pprint
    pprint.pprint(xml_dict)
