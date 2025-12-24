import sys
import os
import torch
import numpy as np
import einops
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from openai import OpenAI  # 必须安装: pip install openai

# 确保能导入项目中的模块
if './' not in sys.path:
    sys.path.append('./')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import DataLoader
# 假设您的 dataset 文件在 src/datasets/dataset_display.py
from src.datasets.dataset_display import MyDataset 

# ==============================================================================
# ⚙️ [配置区域]
# ==============================================================================

# 1. 路径设置
LAYOUT_TXT_PATH = 'demo/txttest'        # 您的 txt 布局文件夹路径
CKPT_PATH = './ckpt/last1.ckpt'     # 您的模型权重路径
CONFIG_PATH = 'configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml' # 配置文件路径
OUTPUT_DIR = './demo/output_llm_clean2'   # 结果保存路径

# 2. LLM 设置 (大模型 API)
API_KEY = "sk-0b6123d9da0a4a2ab04eac5b3d3cf04f"      # 🔴 请替换为您的 API Key
API_BASE_URL = "https://api.deepseek.com"    # DeepSeek 或 OpenAI 地址
API_MODEL_NAME = "deepseek-chat"             
ENABLE_LLM = True                            

# 3. 生成设置
RESOLUTION = 512
BATCH_SIZE = 1       
NUM_SAMPLES = 1      
DDIM_STEPS = 50      
GUIDANCE_SCALE = 7.5 

# DIOR 类别映射
ID_TO_CLASS = {
    0: "airplane", 1: "airport", 2: "baseballfield", 3: "basketballcourt",
    4: "bridge", 5: "chimney", 6: "dam", 7: "Expressway-Service-area",
    8: "Expressway-toll-station", 9: "golffield", 10: "groundtrackfield",
    11: "harbor", 12: "overpass", 13: "ship", 14: "stadium",
    15: "storagetank", 16: "tenniscourt", 17: "trainstation",
    18: "vehicle", 19: "windmill"
}

# ==============================================================================
# 🧠 [核心逻辑] 极简版 LLM Prompt 生成
# ==============================================================================

client = None
if ENABLE_LLM and "sk-" in API_KEY:
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        print("✅ LLM Client 初始化成功")
    except Exception as e:
        print(f"⚠️ LLM 初始化失败: {e}")

def get_objects_from_txt_file(txt_path):
    """读取 txt 文件，返回物体列表"""
    if not os.path.exists(txt_path):
        return []
    objects = []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                parts = line.strip().split()
                class_id = int(parts[0])
                class_name = ID_TO_CLASS.get(class_id, "object")
                objects.append(class_name)
            except:
                continue
    return objects

def generate_clean_prompt(filename, original_txt_path):
    """
    使用 LLM 生成极简、准确的描述
    """
    full_path = os.path.join(original_txt_path, filename)
    object_list = get_objects_from_txt_file(full_path)
    
    # 统计物体 (例如: 2 airplanes, 1 airport)
    from collections import Counter
    counts = Counter(object_list)
    scene_desc = ", ".join([f"{v} {k}" for k, v in counts.items()])
    
    if not scene_desc:
        scene_desc = "background only"

    # --- 关键修改：System Prompt ---
    # 强制 LLM 只做“统计员”和“背景推理员”，禁止做“文学家”
    if ENABLE_LLM and client:
        system_prompt = """
        You are a strict data formatter for satellite images.
        Input: A list of objects and their counts.
        Task: 
        1. Identify the most logical background context (e.g., 'airport' for airplanes, 'harbor' for ships).
        2. Generate a simple sentence.
        
        Strict Constraints:
        - Format MUST be: "An aerial image containing [Quantity] [Objects] with [Context] background."
        - DO NOT use adjectives like 'detailed', 'cinematic', '4k', 'sharp', 'beautiful'.
        - DO NOT add any extra explanation.
        """
        
        user_content = f"Objects found: {scene_desc}."
        
        try:
            response = client.chat.completions.create(
                model=API_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1, # 低温度，保证回答稳定死板
                max_tokens=50
            )
            llm_prompt = response.choices[0].message.content.strip()
            # print(f"🤖 LLM Clean Prompt: {llm_prompt}") 
            return llm_prompt
        except Exception as e:
            print(f"⚠️ LLM 出错: {e}，使用备用规则。")

    # --- 备用规则 (如果 LLM 挂了) ---
    return f"An aerial image containing {scene_desc}."

# ==============================================================================
# 🚀 [主程序]
# ==============================================================================

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def main():
    config = OmegaConf.load(CONFIG_PATH)
    model = load_model_from_config(config, CKPT_PATH)
    sampler = DDIMSampler(model)

    dataset = MyDataset(LAYOUT_TXT_PATH, RESOLUTION, mask_size=64) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n🚀 开始推理 (LLM 极简模式)...")
    
    for batch in tqdm(dataloader, desc="Generating"):
        filenames = batch['filename']
        
        # 1. LLM 生成 Prompt
        refined_prompts = []
        for fname in filenames:
            p = generate_clean_prompt(fname, LAYOUT_TXT_PATH)
            refined_prompts.append(p)
            
        c_prompts = []
        for p in refined_prompts:
            c_prompts.extend([p] * NUM_SAMPLES)

        with torch.no_grad():
            mask_controls = batch['mask_conditions'].float().cuda()
            category_controls = batch['category_conditions'].float().cuda()
            bbox_controls = batch['bboxes'].float().cuda()
            mask_vectors = batch['mask_vector'].float().cuda()
            
            c_bbox = torch.cat([bbox_controls for _ in range(NUM_SAMPLES)], dim=0)
            c_cat = torch.cat([category_controls for _ in range(NUM_SAMPLES)], dim=0)
            c_mask = torch.cat([mask_controls for _ in range(NUM_SAMPLES)], dim=0)
            c_vec = torch.cat([mask_vectors for _ in range(NUM_SAMPLES)], dim=0)

            cond = {
                "c_crossattn": [model.get_learned_conditioning(c_prompts)],
                "bbox_control": [c_bbox],
                "category_control": [c_cat],
                "mask_control": [c_mask],
                "mask_vector": [c_vec],
            }
            
            # 负面提示词：依然保留，保证画面干净，但去掉了风格化的负面词
            neg_text = "low quality, blur, pixelated, distortion, lowres, bad anatomy, text, watermark, foggy"
            un_cond = {
                "c_crossattn": [model.get_learned_conditioning([neg_text] * len(c_prompts))],
                "bbox_control": [c_bbox],
                "category_control": [c_cat],
                "mask_control": [c_mask],
                "mask_vector": [c_vec],
            }

            shape = (4, RESOLUTION // 8, RESOLUTION // 8)
            samples, _ = sampler.sample(
                S=DDIM_STEPS,
                conditioning=cond,
                batch_size=len(c_prompts),
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=GUIDANCE_SCALE,
                unconditional_conditioning=un_cond,
                eta=0.0
            )

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            
            for i, filename in enumerate(filenames):
                for j in range(NUM_SAMPLES):
                    img_tensor = x_samples[i * NUM_SAMPLES + j]
                    img_np = 255. * einops.rearrange(img_tensor, 'c h w -> h w c').cpu().numpy()
                    img_pil = Image.fromarray(img_np.astype(np.uint8))
                    
                    base_name = filename.replace('.txt', '').replace('txt', '')
                    save_name = f"{base_name}.jpg" if NUM_SAMPLES == 1 else f"{base_name}_{j}.jpg"
                    
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    img_pil.save(save_path)
                    
                    # (建议) 把 Prompt 保存下来看看 LLM 听不听话
                    with open(save_path.replace('.jpg', '.txt'), 'w') as f:
                        f.write(refined_prompts[i])

    print("\n✅ 完成！")

if __name__ == "__main__":
    main()