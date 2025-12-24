import os
import xml.etree.ElementTree as ET
import tqdm

# --- 配置 ---
# DIOR 数据集的根目录 (请根据你的实际路径修改)
DIOR_ROOT = "./datasets/DIOR-VOC" 
# 提取哪一部分的数据？通常 FID 测的是 'test' 集
SET_NAME = "test"  
# 输出文件夹
OUTPUT_DIR = "./test_layouts_txt"

# DIOR 类别 ID 映射 (必须和你训练时的 dataset.py 一致)
CLASS_MAP = {
    "airplane": 0, "airport": 1, "baseballfield": 2, "basketballcourt": 3, 
    "bridge": 4, "chimney": 5, "dam": 6, "Expressway-Service-area": 7, 
    "Expressway-toll-station": 8, "golffield": 9, "groundtrackfield": 10, 
    "harbor": 11, "overpass": 12, "ship": 13, "stadium": 14, 
    "storagetank": 15, "tenniscourt": 16, "trainstation": 17, 
    "vehicle": 18, "windmill": 19
}

def parse_xml_to_txt(xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图片尺寸用于归一化
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    lines = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[name]
        
        # 获取 OBB 坐标 (robndbox)
        robndbox = obj.find('robndbox')
        if robndbox is not None:
            # 提取 x1, y1, x2, y2, x3, y3, x4, y4
            coords = [
                'x_left_top', 'y_left_top', 'x_right_top', 'y_right_top',
                'x_right_bottom', 'y_right_bottom', 'x_left_bottom', 'y_left_bottom'
            ]
            points = []
            try:
                for k in coords:
                    val = float(robndbox.find(k).text)
                    # 归一化到 0-1
                    if 'x_' in k:
                        points.append(val / width)
                    else:
                        points.append(val / height)
                
                # 格式化为字符串: class_id p1 p2 ... p8
                line_str = f"{class_id} " + " ".join([f"{p:.6f}" for p in points])
                lines.append(line_str)
            except:
                pass # 忽略坏数据

    # 写入 txt 文件
    if lines:
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

def main():
    # 1. 读取测试集列表
    txt_list_path = os.path.join(DIOR_ROOT, "VOC2007", "ImageSets", "Main", f"{SET_NAME}.txt")
    if not os.path.exists(txt_list_path):
        print(f"Error: 找不到列表文件 {txt_list_path}")
        return

    with open(txt_list_path, 'r') as f:
        file_ids = [x.strip() for x in f.readlines()]

    print(f"正在处理 {len(file_ids)} 张测试图片...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 遍历并转换
    for file_id in tqdm.tqdm(file_ids):
        xml_path = os.path.join(DIOR_ROOT, "Annotations", "Oriented_Bounding_Boxes", f"{file_id}.xml")
        output_txt_path = os.path.join(OUTPUT_DIR, f"{file_id}.txt")
        
        if os.path.exists(xml_path):
            parse_xml_to_txt(xml_path, output_txt_path)

    print(f"完成！所有布局文件已保存在: {OUTPUT_DIR}")
    print("你可以让你的推理脚本读取这个文件夹里的文件来批量生成图片了。")

if __name__ == "__main__":
    main()