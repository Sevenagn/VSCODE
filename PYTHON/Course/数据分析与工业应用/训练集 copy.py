import os
import pandas as pd
from PIL import Image

def resize_and_grayscale_image(image_path, size=(28, 28)):
    with Image.open(image_path) as img:
        # 转换为灰度图像
        img = img.convert('L')
        # 缩放图像
        img = img.resize(size, Image.Resampling.LANCZOS)
        return list(img.getdata())

def convert_dataset_to_csv(image_dir, csv_file):
    data = []
    for class_dir in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_dir)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 读取并灰度化、缩放图像数据
                    image_data = resize_and_grayscale_image(image_path)
                    # 将标签放在数据的开头
                    data.append([class_dir] + image_data)
    
    # 将数据转换为DataFrame
    max_pixels = 28 * 28  # 每张图像的像素数量
    df = pd.DataFrame(data, columns=['Label'] + [f'Pixel{i+1}' for i in range(max_pixels)])
    df.to_csv(csv_file, index=False)

# 使用示例
image_dir = r'C:\Users\seven.zhou\Downloads\raw-img'  # 图像数据集目录
csv_file = 'dataset.csv'  # 输出的CSV文件
convert_dataset_to_csv(image_dir, csv_file)
