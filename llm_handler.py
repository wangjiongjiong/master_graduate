import requests

class DeepSeekClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.deepseek.com/chat/completions"

    def fetch_rich_caption(self, class_names):
        items = ", ".join(class_names)
        # 专门针对 DIOR 遥感影像的提示词
        prompt = (f"As an expert in satellite imagery, describe a scene containing: {items}. "
                  f"Provide a natural, vivid one-sentence description focusing on typical aerial textures "
                  f"and spatial arrangements. Keep it under 40 words.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=15)
            res_json = response.json()
            
            # 如果报错 'choices'，我们就看看到底返回了什么
            if 'choices' not in res_json:
                print(f">>> DeepSeek API Response Error: {res_json}")
                return None
                
            return res_json['choices'][0]['message']['content']
        except Exception as e:
            print(f">>> Network or JSON Error: {e}")
            return None