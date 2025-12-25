import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
# 你的原始数据集路径
VOC_ROOT = "/hy-tmp/wjm/master_graduate/datasets/DIOR-VOC"
XML_FOLDER = os.path.join(VOC_ROOT, "Annotations/Horizontal_Bounding_Boxes")
IMG_FOLDER = os.path.join(VOC_ROOT, "VOC2007/JPEGImages")
# 用于划分训练集和验证集的 txt 文件路径
SETS_FOLDER = os.path.join(VOC_ROOT, "VOC2007/ImageSets/Main")

# 输出的新数据集路径 (会自动创建)
OUT_ROOT = "/hy-tmp/wjm/master_graduate/datasets/DIOR-YOLO"

# 确保输出目录存在
for folder in ['train', 'val']:
    os.makedirs(os.path.join(OUT_ROOT, f'images/{folder}'), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, f'labels/{folder}'), exist_ok=True)

# DIOR 的 20 个类别 (顺序必须固定)
CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge", 
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", 
    "golffield", "groundtrackfield", "harbor", "overpass", "ship", 
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"
]

def convert(size, box):
    """将 (xmin, xmax, ymin, ymax) 转换为 (x, y, w, h) 并归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_set(image_set_file, mode):
    """处理训练集或验证集"""
    if not os.path.exists(image_set_file):
        print(f"Warning: {image_set_file} 不存在，跳过。")
        return

    with open(image_set_file, 'r') as f:
        ids = f.read().strip().split()
    
    print(f"正在处理 {mode} 集 (共 {len(ids)} 张)...")
    
    for image_id in tqdm(ids):
        image_id = image_id.strip()
        if not image_id: continue

        # 1. 复制图片
        src_img = os.path.join(IMG_FOLDER, f"{image_id}.jpg")
        dst_img = os.path.join(OUT_ROOT, "images", mode, f"{image_id}.jpg")
        
        # 如果找不到图，就跳过
        if not os.path.exists(src_img): 
            continue
        shutil.copy(src_img, dst_img)
        
        # 2. 转换标签 (XML -> TXT)
        xml_file = os.path.join(XML_FOLDER, f"{image_id}.xml")
        if not os.path.exists(xml_file): continue
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            txt_out_path = os.path.join(OUT_ROOT, "labels", mode, f"{image_id}.txt")
            
            with open(txt_out_path, 'w') as f_out:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in CLASSES: continue
                    cls_id = CLASSES.index(cls)
                    xmlbox = obj.find('bndbox')
                    
                    # 读取 XML 坐标
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    
                    # 转换格式
                    bb = convert((w, h), b)
                    f_out.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
        except Exception as e:
            print(f"Error converting {image_id}: {e}")

if __name__ == "__main__":
    # 处理训练集
    process_set(os.path.join(SETS_FOLDER, "train.txt"), "train")
    # 处理验证集 (如果没有 val.txt，可以用 test.txt 代替)
    process_set(os.path.join(SETS_FOLDER, "test.txt"), "val")
    print("转换完成！新数据集在: ./datasets/DIOR-YOLO")