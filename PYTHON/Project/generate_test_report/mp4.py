import imageio
import logging

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

def convert_to_mp4(input_file, output_file):
    reader = imageio.get_reader(input_file)
    fps = reader.get_meta_data()['fps']
    logging.debug(f"FPS: {fps}")

    writer = imageio.get_writer(output_file, fps=fps)

    # 获取 pts_delta
    meta_data = reader.get_meta_data()
    pts_delta = meta_data.get('pts_delta')

    if pts_delta is None or not isinstance(pts_delta, float):
        logging.warning(f"Invalid pts_delta value: {pts_delta}. Defaulting to 1.0.")
        pts_delta = 1.0

    for index, frame in enumerate(reader):
        if index is None:
            logging.warning(f"Frame at index {index} has None value. Skipping.")
            continue

        index_pts = int(index * pts_delta)
        # 其他处理

        writer.append_data(frame)

    writer.close()

# 使用示例
input_file = r'C:\Users\17167\OneDrive\桌面\监控\1.264'
output_file = r'C:\Users\17167\OneDrive\桌面\监控\1.mp4'

try:
    convert_to_mp4(input_file, output_file)
except Exception as e:
    print(f"Error: {e}")
