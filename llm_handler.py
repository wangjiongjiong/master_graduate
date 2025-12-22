import json
import requests
import time

class DeepSeekClient:
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url

    def call_api(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}, # 强制返回JSON
            "stream": False
        }
        try:
            response = requests.post(f"{self.base_url}/v1/chat/completions", headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Connection Error: {e}")
            return None

    def fetch_evolved_descriptions(self, class_list, version=0):
        # 演进阶段的主题定义
        themes = [
            "Focus on high-resolution satellite sensor characteristics, sharp textures, and clear object outlines.",
            "Focus on realistic lighting, solar reflections on metallic surfaces, and accurate ground shadows.",
            "Focus on intricate structural details of remote sensing targets and atmospheric clarity.",
            "Focus on professional aerial photography aesthetics, emphasizing contrast and geometric precision."
        ]
        current_theme = themes[version % len(themes)]

        prompt = f"""
        You are an expert in remote sensing imagery. Please generate enhanced descriptions for the 20 categories in the DIOR dataset.
        
        [Current Evolution Goal]: {current_theme}
        
        [Strict Rules]:
        1. Must include the exact category name in each description.
        2. AVOID specific backgrounds like 'desert', 'grassland', or 'city' to prevent conflict with actual image pixels.
        3. Use keywords: 'high-resolution', 'sharp texture', 'solar reflections', 'clear shadows', 'top-down view'.
        4. Keep each description under 15 words.
        5. Return ONLY a JSON object: {{"category_name": "enhanced description"}}
        
        Categories to process: {class_list}
        """
        
        response_str = self.call_api(prompt)
        if response_str:
            try:
                # 去除可能的Markdown标记
                clean_json = response_str.strip('`').replace('json\n', '')
                return json.loads(clean_json)
            except Exception as e:
                print(f"JSON Parsing Error: {e}")
                return None
        return None