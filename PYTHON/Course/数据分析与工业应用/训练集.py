import numpy as np
import pandas as pd

def generate_training_set(file_name, num_samples=1000):
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 生成随机的标签数据（0, 1, 2, 3四个类别）
    labels = np.random.randint(0, 9, num_samples)

    # 生成随机的图像数据，每个样本包含 28x28 = 784 个像素值（0-255之间的整数）
    images = np.random.randint(0, 256, (num_samples, 784))

    # 将标签和图像数据合并为一个 DataFrame
    df = pd.DataFrame(data=np.column_stack((labels, images)))

    # 保存为 CSV 文件
    df.to_csv(file_name, index=False, header=False)
    print(f"Training set saved to {file_name}")

if __name__ == "__main__":
    generate_training_set("2train.csv", num_samples=1000)
