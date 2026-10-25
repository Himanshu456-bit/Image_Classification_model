"""
Helper script to generate a few sample images in category folders.
This allows the user to test adding category folders immediately!
"""
import os
from PIL import Image, ImageDraw

def create_samples():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_dataset")

    categories = {
        "Red_Apples": (220, 20, 60),
        "Green_Pears": (34, 139, 34),
        "Blue_Berries": (30, 144, 255)
    }

    for cat_name, color in categories.items():
        cat_dir = os.path.join(base_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        for i in range(3):
            img = Image.new("RGB", (200, 200), (245, 245, 245))
            draw = ImageDraw.Draw(img)
            # Draw circle with offset to give variation
            offset = i * 10
            draw.ellipse([40 + offset, 40 + offset, 160 - offset, 160 - offset], fill=color)
            img.save(os.path.join(cat_dir, f"sample_{i+1}.png"))

    print(f"Sample category folders generated in: {base_dir}")

if __name__ == "__main__":
    create_samples()
