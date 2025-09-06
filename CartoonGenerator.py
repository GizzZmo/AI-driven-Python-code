import requests
import json
import base64
from PIL import Image
import io

class CartoonGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.gemini.com/v1/"
        
    def generate_cartoon(self, prompt, page_count=12, strips_per_page=4):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(
            f"{self.base_url}generate",
            headers=headers,
            data=json.dumps({
                "prompt": prompt,
                "page_count": page_count,
                "strips_per_page": strips_per_page
            })
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error generating cartoon: {response.text}")
            
    def save_pages(self, cartoon_data, output_dir):
        for i, page in enumerate(cartoon_data["pages"]):
            image_data = base64.b64decode(page["image"])
            image = Image.open(io.BytesIO(image_data))
            
            # Save front side
            front_side = image.crop((0, 0, image.width//2, image.height))
            front_side.save(f"{output_dir}/page_{i+1}_front.png")
            
            # Save back side
            back_side = image.crop((image.width//2, 0, image.width, image.height))
            back_side.save(f"{output_dir}/page_{i+1}_back.png")

# Example usage
api_key = input("Enter your Gemini API key: ")
generator = CartoonGenerator(api_key)

cartoon_data = generator.generate_cartoon(
    prompt="A day in the life of a developer",
    page_count=12,
    strips_per_page=4
)

generator.save_pages(cartoon_data, "./output")
