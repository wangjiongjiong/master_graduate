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
# ⚙️ [配置区域] 请在这里修改您的路径和 Key
# ==============================================================================

# 1. 路径设置
LAYOUT_TXT_PATH = 'demo/txttest'   # 您的 txt 布局文件夹路径
CKPT_PATH = './ckpt/last1.ckpt'             # 您的模型权重路径
#CKPT_PATH = './ckpt/aerogen_diorr_last.ckpt' 
CONFIG_PATH = 'configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml' # 配置文件路径
OUTPUT_DIR = './demo/output_llm_enhanced'   # 结果保存路径

# 2. LLM 设置 (大模型 API)
# 推荐使用 DeepSeek (便宜且强) 或 OpenAI
API_KEY = "sk-0b6123d9da0a4a2ab04eac5b3d3cf04f"  # 🔴 请替换为您的 API Key
API_BASE_URL = "https://api.deepseek.com"      # 如果用 OpenAI，请删掉这行或留空
API_MODEL_NAME = "deepseek-chat"               # 模型名称，例如 "gpt-4o" 或 "deepseek-chat"
ENABLE_LLM = True                              # 如果不想用 LLM，改为 False，将使用规则拼接

# 3. 生成设置
RESOLUTION = 512
BATCH_SIZE = 1        # 建议为 1，因为 LLM 生成 Prompt 需要时间
NUM_SAMPLES = 1      # 每张布局生成几张图 (FID 测试建议 1)
DDIM_STEPS = 50       # 采样步数
GUIDANCE_SCALE = 7.5  # CFG Scale (越大越听话，越小越自然)

# DIOR 类别映射 (0-19)
ID_TO_CLASS = {
    0: "airplane", 1: "airport", 2: "baseballfield", 3: "basketballcourt",
    4: "bridge", 5: "chimney", 6: "dam", 7: "Expressway-Service-area",
    8: "Expressway-toll-station", 9: "golffield", 10: "groundtrackfield",
    11: "harbor", 12: "overpass", 13: "ship", 14: "stadium",
    15: "storagetank", 16: "tenniscourt", 17: "trainstation",
    18: "vehicle", 19: "windmill"
}

# ==============================================================================
# 🧠 [核心逻辑] Prompt 生成模块
# ==============================================================================

# 初始化 LLM 客户端
client = None
if ENABLE_LLM and "sk-" in API_KEY:
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        print("✅ LLM Client 初始化成功")
    except Exception as e:
        print(f"⚠️ LLM 初始化失败: {e}，将使用备用规则模式。")

def get_objects_from_txt_file(txt_path):
    """读取 txt 文件，分析里面有什么物体"""
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

def generate_enhanced_prompt(filename, original_txt_path):
    """
    输入: 文件名 (例如 '00001.txt') 和 完整路径
    输出: 增强后的 Prompt
    """
    full_path = os.path.join(original_txt_path, filename)
    object_list = get_objects_from_txt_file(full_path)
    
    # 统计物体数量，例如: "2 airplanes, 1 groundtrackfield"
    from collections import Counter
    counts = Counter(object_list)
    scene_desc = ", ".join([f"{v} {k}(s)" for k, v in counts.items()])
    
    if not scene_desc:
        scene_desc = "various objects"

    # --- 模式 A: 使用 LLM 生成 (高质量) ---
    if ENABLE_LLM and client:
        system_prompt = """
        You are an expert in writing prompts for AI satellite image generation.
        Input: A list of objects in the scene.
        Output: A concise, photorealistic, high-quality caption for Stable Diffusion.
        
        Rules:
        1. Start with "A high-resolution top-down satellite image of..."
        2. Describe the background texture (e.g., concrete for planes, blue water for ships, red track for sports).
        3. Add quality keywords: "8k, sharp focus, cinematic lighting, detailed shadows, hdr".
        4. Keep it under 60 words. No explanations.
        """
        user_content = f"Scene objects: {scene_desc}."
        
        try:
            response = client.chat.completions.create(
                model=API_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=100
            )
            llm_prompt = response.choices[0].message.content.strip()
            # print(f"🤖 LLM Prompt: {llm_prompt}") 
            return llm_prompt
        except Exception as e:
            print(f"⚠️ LLM 调用出错: {e}，切换回规则模式。")

    # --- 模式 B: 规则拼接 (备用/快速) ---
    # 如果没开 LLM 或者调用失败，用这个
    base = f"A professional high-resolution optical satellite imagery, top-down view of {scene_desc}. "
    
    details = ""
    if "airplane" in scene_desc: details += "Parked on grey concrete apron with markings. "
    elif "ship" in scene_desc: details += "Docked in deep blue water with waves. "
    elif "groundtrackfield" in scene_desc: details += "Red running track with green grass field. "
    elif "tenniscourt" in scene_desc: details += "Blue and green hard court surfaces. "
    elif "storagetank" in scene_desc: details += "White industrial tanks structure. "
    
    suffix = "Highly detailed, 4k resolution, sharp focus, cinematic lighting, realistic textures, clear shadows."
    return base + details + suffix

# ==============================================================================
# 🚀 [主程序] 推理循环
# ==============================================================================

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd)
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

def main():
    # 1. 准备模型
    config = OmegaConf.load(CONFIG_PATH)
    model = load_model_from_config(config, CKPT_PATH)
    sampler = DDIMSampler(model)

    # 2. 准备数据
    dataset = MyDataset(LAYOUT_TXT_PATH, RESOLUTION, mask_size=64) # 注意 mask_size 需根据训练配置调整
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # Windows下建议设0

    print(f"\n🚀 开始推理...")
    print(f"📂 输入路径: {LAYOUT_TXT_PATH}")
    print(f"💾 输出路径: {OUTPUT_DIR}")
    print(f"🤖 LLM 启用状态: {ENABLE_LLM}")

    # 3. 循环生成
    for batch in tqdm(dataloader, desc="Generating"):
        filenames = batch['filename']
        
        # --- [关键步骤] 生成增强的 Prompts ---
        refined_prompts = []
        for fname in filenames:
            # 传入文件名，去 txt 里查物体，然后让 LLM 写 prompt
            p = generate_enhanced_prompt(fname, LAYOUT_TXT_PATH)
            refined_prompts.append(p)
        
        # 批量复制 Prompt (如果 num_samples > 1)
        # 例如: ["prompt1"] -> ["prompt1", "prompt1"]
        c_prompts = []
        for p in refined_prompts:
            c_prompts.extend([p] * NUM_SAMPLES)

        with torch.no_grad():
            # 准备 Condition
            mask_controls = batch['mask_conditions'].float().cuda()
            category_controls = batch['category_conditions'].float().cuda()
            bbox_controls = batch['bboxes'].float().cuda()
            mask_vectors = batch['mask_vector'].float().cuda()
            
            # 堆叠 Condition (支持 num_samples)
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
            
            # --- [关键步骤] 负面提示词 (Negative Prompt) ---
            neg_prompt_text = "low quality, blur, pixelated, distortion, lowres, bad anatomy, text, watermark, foggy, haze, cartoon, painting, illustration"
            un_cond = {
                "c_crossattn": [model.get_learned_conditioning([neg_prompt_text] * len(c_prompts))],
                "bbox_control": [c_bbox],
                "category_control": [c_cat],
                "mask_control": [c_mask],
                "mask_vector": [c_vec],
            }

            # 采样
            shape = (4, RESOLUTION // 8, RESOLUTION // 8)
            samples, _ = sampler.sample(
                S=DDIM_STEPS,
                conditioning=cond,
                batch_size=len(c_prompts),
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=GUIDANCE_SCALE,
                unconditional_conditioning=un_cond,
                eta=0.2 # 增加一点随机性
            )

            # 解码 VAE
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            
            # 保存图片
            for i, filename in enumerate(filenames):
                # 每个 layout 可能生成多张
                for j in range(NUM_SAMPLES):
                    img_tensor = x_samples[i * NUM_SAMPLES + j]
                    img_np = 255. * einops.rearrange(img_tensor, 'c h w -> h w c').cpu().numpy()
                    img_pil = Image.fromarray(img_np.astype(np.uint8))
                    
                    # 保存结构: output_dir/00001.jpg
                    # 如果 num_samples > 1, 可以在文件名加后缀
                    save_name = filename.replace('.txt', '.jpg').replace('txt', 'jpg')
                    if NUM_SAMPLES > 1:
                        save_name = f"{filename.split('.')[0]}_{j}.jpg"
                    
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    img_pil.save(save_path)
                    
                    # 如果想保存对应的 Prompt 方便检查
                    with open(save_path.replace('.jpg', '.txt'), 'w') as f:
                        f.write(refined_prompts[i])

    print("\n✅ 全部完成！")

if __name__ == "__main__":
    main()