'''
Created by Han Xu
email:736946693@qq.com
'''
import xml.etree.ElementTree as ET


def read_configuration(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取camvid数据集路径
    camvid_path = root.find('camvid_path').text

    # 获取模型路径
    HANet_1HAM_path = root.find('HANet_1HAM_path').text
    HANet_2HAM_path = root.find('HANet_2HAM_path').text
    HANet_3HAM_path = root.find('HANet_3HAM_path').text
    HANet_4HAM_path = root.find('HANet_4HAM_path').text

    # 获取保存模型的路径
    save_path = root.find('save_path').text

    return {"camvid_path":camvid_path,
            "HANet_1HAM_path":HANet_1HAM_path,
            "HANet_2HAM_path":HANet_2HAM_path,
            "HANet_3HAM_path":HANet_3HAM_path,
            "HANet_4HAM_path":HANet_4HAM_path,
            "save_path":save_path}


# 用于测试的 XML 文件路径
xml_file_path = "conf.xml"

# 读取配置信息
import os

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 切换到该目录
os.chdir(current_directory)

xml_info= read_configuration(xml_file_path)
camvid_path=xml_info["camvid_path"]
HANet_1HAM_path=xml_info["HANet_1HAM_path"]
HANet_2HAM_path=xml_info["HANet_2HAM_path"]
HANet_3HAM_path=xml_info["HANet_3HAM_path"]
HANet_4HAM_path=xml_info["HANet_4HAM_path"]
save_path=xml_info["save_path"]

if __name__ == "__main__":
    # 打印读取到的信息
    print(xml_info)
