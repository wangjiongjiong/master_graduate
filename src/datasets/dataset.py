import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

max_length = 15

# --- 原有的辅助函数保持不变 ---
def resize_image_and_bboxes(image, bboxes, target_size=(512, 512)):
    original_size = (800,800)
    image_resized = cv2.resize(image, target_size)
    scale_x = target_size[1] / original_size[1]
    scale_y = target_size[0] / original_size[0]
    resized_bboxes = []
    for bbox in bboxes:
        x_coords = np.array(bbox[0]) * scale_x
        y_coords = np.array(bbox[1]) * scale_y
        resized_bboxes.append((x_coords, y_coords))
    return image_resized, resized_bboxes

def random_rotation(image, bboxes, degrees):
    height, width = image.shape[:2]
    if degrees == 0: return image, bboxes
    M = cv2.getRotationMatrix2D((width // 2, height // 2), degrees, 1.0)
    cos_angle, sin_angle = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_width = int(height * sin_angle + width * cos_angle)
    new_height = int(height * cos_angle + width * sin_angle)
    M[0, 2] += (new_width / 2) - (width / 2)
    M[1, 2] += (new_height / 2) - (height / 2)
    image_rot = cv2.warpAffine(image, M, (new_width, new_height))
    new_bboxes = []
    for bbox in bboxes:
        points = np.array([[x, y] for x, y in zip(bbox[0], bbox[1])])
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = M.dot(points_ones.T).T
        new_bboxes.append((transformed_points[:, 0], transformed_points[:, 1]))
    return image_rot, new_bboxes

def rotate_point(px, py, cx, cy, angle):
    rad = np.radians(angle)
    x_shifted, y_shifted = px - cx, py - cy
    new_x = x_shifted * np.cos(rad) - y_shifted * np.sin(rad)
    new_y = x_shifted * np.sin(rad) + y_shifted * np.cos(rad)
    return new_x + cx, new_y + cy

def calculate_max_square(corners):
    x_coords, y_coords = corners[:, 0], corners[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    square_size = int(min(x_max - x_min, y_max - y_min))
    square_center_x, square_center_y = int((x_min + x_max) // 2), int((y_min + y_max) // 2)
    x_min_square = max(0, square_center_x - square_size // 2)
    y_min_square = max(0, square_center_y - square_size // 2)
    return x_min_square, y_min_square, x_min_square + square_size, y_min_square + square_size

def crop_and_resize(image, corners, target_size=(512, 512)):
    x_min_s, y_min_s, x_max_s, y_max_s = calculate_max_square(corners)
    image_cropped = image[int(y_min_s):int(y_max_s), int(x_min_s):int(x_max_s)]
    if image_cropped.size == 0: raise ValueError("Cropped image is empty.")
    return cv2.resize(image_cropped, target_size)

# --- 修改后的 MyDataset 类 ---
class MyDataset(Dataset):
    def __init__(self, original_path, image_file_path, resolution, mask_size, mode='train', rotate_once=False, rotate_twice=False):
        self.original_path = original_path
        self.image_file_path = image_file_path
        self.resolution = (resolution, resolution)
        self.mask_size = (mask_size, mask_size)
        self.rotate_once = rotate_once
        self.rotate_twice = rotate_twice
        
        # --- 新增：DeepSeek 缓存字典 ---
        self.llm_cache = {} 
        
        self.category = {
            "airplane": 0, "airport": 1, "baseballfield": 2, "basketballcourt": 3, "bridge": 4,
            "chimney": 5, "dam": 6, "Expressway-Service-area": 7, "Expressway-toll-station": 8,
            "golffield": 9, "groundtrackfield": 10, "harbor": 11, "overpass": 12, "ship": 13,
            "stadium": 14, "storagetank": 15, "tenniscourt": 16, "trainstation": 17, "vehicle": 18,
            "windmill": 19
        }
        self.descriptions = {
            "airplane": "airplane parked on the ground", "airport": "busy airport",
            "baseballfield": "green baseball field", "basketballcourt": "outdoor basketball court",
            "bridge": "long bridge", "chimney": "tall chimney", "dam": "large dam",
            "Expressway-Service-area": "crowded expressway service area",
            "Expressway-toll-station": "busy expressway toll station",
            "golffield": "well-maintained golf field", "groundtrackfield": "athletic ground track field",
            "harbor": "bustling harbor", "overpass": "elevated overpass", "ship": "ship on the water",
            "stadium": "large stadium", "storagetank": "industrial storage tank",
            "tenniscourt": "clay tennis court", "trainstation": "crowded train station",
            "vehicle": "vehicle on the road", "windmill": "rotating windmill"
        }
        self.category_embeddings = np.load('./datasets/category_embeddings.npy')
        self.condition_dropout_prob = 0.1
        self.mode = mode

        txt_file_path = f'./datasets/DIOR-VOC/VOC2007/ImageSets/Main/{"train" if mode=="aug_data" else mode}.txt'
        with open(txt_file_path, 'r') as file:
            file_names = file.read().splitlines()
        
        self.files = [os.path.join(original_path, f'{file_name}.xml') for file_name in file_names]

    # --- 新增：更新接口 ---
    def update_llm_cache(self, new_captions):
        """用于 main.py 通过 Callback 注入 DeepSeek 的描述"""
        self.llm_cache.update(new_captions)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        xml_file_path = self.files[idx]
        # 获取基础信息，注意 _parse_annotation 保持不变
        image, filename, img_size, bboxes, class_names, _ = self._parse_annotation(xml_file_path)
        
        # --- 修改：传递 filename 给 prompt 生成函数 ---
        prompt = self._generate_prompt(filename, class_names)
        
        # --- 以下原有的数据增强逻辑完全保持不变 ---
        image_resized, bboxes_resized = resize_image_and_bboxes(image, bboxes, target_size=(512, 512))
        
        if self.rotate_once or self.rotate_twice:
            angle = random.choice([0, 90, 180, 270])
            rot_image, rot_bboxes = random_rotation(image_resized, bboxes_resized, angle)
            if self.rotate_twice:
                angle_fine = random.uniform(-15, 15)
                rot_image, rot_bboxes = random_rotation(rot_image, rot_bboxes, angle_fine)
        else:
            rot_image, rot_bboxes = image_resized, bboxes_resized

        if self.rotate_twice:
            h, w = rot_image.shape[:2]
            corners = np.array([[0, 0], [0, h], [w, 0], [w, h]])
            # 注意：此处旋转中心坐标逻辑保持你的原有实现
            corners_rot = np.array([rotate_point(x, y, w // 2, h // 2, angle_fine if self.rotate_twice else 0) for x, y in corners])
            rot_image = crop_and_resize(rot_image, corners_rot, target_size=(512, 512))
        
        mask_conditions, final_bboxes, mask_vector = self._generate_mask(rot_bboxes, self.resolution, self.mask_size)
        category_conditions = self._get_category_conditions(class_names)
        
        return dict(
            jpg=rot_image,
            filename=filename,
            txt=prompt,
            mask_conditions=mask_conditions,
            bboxes=final_bboxes,
            category_conditions=category_conditions,
            mask_vector=mask_vector
        )

    # --- 修改：Prompt 生成逻辑 ---
    def _generate_prompt(self, filename, class_names):
        if self.mode == 'train':
            if random.random() < self.condition_dropout_prob:
                return ''

        # 优先从缓存读取 DeepSeek 描述
        if filename in self.llm_cache:
            return self.llm_cache[filename]

        # 如果没有缓存，运行原来的模板拼接逻辑
        unique_class_names = list(set(class_names))
        prompt = 'an aerial image with '
        if len(unique_class_names) == 1:
            prompt += self.descriptions[unique_class_names[0]]
        else:
            prompt += ', '.join([self.descriptions[name] for name in unique_class_names[:-1]]) + ' and ' + self.descriptions[unique_class_names[-1]]
        return prompt

    # --- 原有方法保持不变 ---
    def _generate_mask(self, bboxes, source_size, target_size):
        source_height, source_width = (512, 512)
        target_height, target_width = target_size
        mask_list, bbox_list = [], []
        mask_vector = np.zeros(max_length, dtype=np.float32)
        for i, bbox in enumerate(bboxes):
            if i >= max_length: break
            x_coords, y_coords = bbox
            x_scaled = (np.array(x_coords) * target_width / source_width).astype(int)
            y_scaled = (np.array(y_coords) * target_height / source_height).astype(int)
            mask = np.zeros((target_height, target_width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.stack((x_scaled, y_scaled), axis=-1)], 1)
            mask_list.append(mask)
            bbox_list.append(np.stack((x_scaled, y_scaled), axis=-1).flatten())
            mask_vector[i] = 1
        while len(mask_list) < max_length:
            mask_list.append(np.zeros((target_height, target_width), dtype=np.uint8))
            bbox_list.append(np.zeros(8, dtype=int))
        return np.stack(mask_list), np.stack(bbox_list), mask_vector

    def _get_category_conditions(self, class_names):
        category_conditions = []
        for class_name in class_names:
            category_index = self.category[class_name]
            category_conditions.append(self.category_embeddings[category_index])
        if len(category_conditions) > max_length:
            category_conditions = category_conditions[:max_length]
        while len(category_conditions) < max_length:
            category_conditions.append(np.zeros(768))
        return np.stack(category_conditions)

    def _parse_annotation(self, xml_file_path):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        filename = root.find('filename').text
        size = root.find('size')
        img_size = (int(size.find('width').text), int(size.find('height').text), int(size.find('depth').text))
        objects = root.findall('object')
        bboxes, class_names, prompt_parts = [], [], []
        for obj in objects:
            name = obj.find('name').text
            class_names.append(name)
            robndbox = obj.find('robndbox')
            coords = ['x_left_top', 'y_left_top', 'x_right_top', 'y_right_top', 
                      'x_right_bottom', 'y_right_bottom', 'x_left_bottom', 'y_left_bottom']
            v = [int(robndbox.find(k).text) for k in coords]
            bboxes.append(([v[0], v[2], v[4], v[6]], [v[1], v[3], v[5], v[7]]))
            prompt_parts.append(f"{name} at {v}")
        
        image_path = os.path.join(self.image_file_path, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resolution)
        image = (image.astype(np.float32) / 127.5) - 1.0
        return image, filename, img_size, bboxes, class_names, " and ".join(prompt_parts)