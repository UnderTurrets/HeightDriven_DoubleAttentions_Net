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
    camvid_path=root.find('camvid_path').text

    # 获取模型路径
    HANet_oneHAM_path = root.find('HANet_oneHAM_path').text
    HANet_twoHAM_path = root.find('HANet_twoHAM_path').text

    return camvid_path, HANet_oneHAM_path, HANet_twoHAM_path

# 用于测试的 XML 文件路径
xml_file_path = "conf.xml"

# 读取配置信息
import os
# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 切换到该目录
os.chdir(current_directory)

camvid_path, HANet_oneHAM_path, HANet_twoHAM_path = read_configuration(xml_file_path)

if __name__ == "__main__":
# 打印读取到的信息
    print(f"camvid_path: {camvid_path}")
    print(f"HANet_oneHAM_path: {HANet_oneHAM_path}")
    print(f"HANet_twoHAM_path: {HANet_twoHAM_path}")