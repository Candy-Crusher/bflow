import os
import shutil
import csv

# 源目录和目标目录
mode = 'test'
source_dir = f'./DSEC/{mode}'
target_dir = f'./{mode}'

# 遍历源目录中的所有子文件夹
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)

    # 确保是文件夹
    if os.path.isdir(subfolder_path):
        flow_folder = os.path.join(subfolder_path, 'flow')

        # 检查子文件夹中是否包含 flow 文件夹
        if os.path.exists(flow_folder):
            # 创建目标子文件夹
            target_subfolder = os.path.join(target_dir, subfolder)
            # 如果文件夹存在 删除
            if os.path.exists(target_subfolder):
                shutil.rmtree(target_subfolder)
            os.makedirs(target_subfolder, exist_ok=True)

            # 映射 calibration
            calibration_src = os.path.join(subfolder_path, 'calibration')
            calibration_target = os.path.join(target_subfolder, 'calibration')
            if os.path.exists(calibration_src):
                os.symlink(calibration_src, calibration_target)

            # 映射 events/left
            events_left_src = os.path.join(subfolder_path, 'events', 'left')
            events_left_target = os.path.join(target_subfolder, 'events', 'left')
            if os.path.exists(events_left_src):
                os.makedirs(os.path.dirname(events_left_target), exist_ok=True)
                os.symlink(events_left_src, events_left_target)

            # 映射 flow
            flow_src = os.path.join(subfolder_path, 'flow', 'forward')
            flow_target = os.path.join(target_subfolder, 'flow', 'forward')
            os.makedirs(os.path.dirname(flow_target), exist_ok=True)
            if os.path.exists(flow_src):
                os.symlink(flow_src, flow_target)

            # 映射 images/timestamps.txt
            timestamps_src = os.path.join(subfolder_path, 'images', 'timestamps.txt')
            timestamps_target = os.path.join(target_subfolder, 'images', 'timestamps.txt')
            if os.path.exists(timestamps_src):
                os.makedirs(os.path.dirname(timestamps_target), exist_ok=True)
                os.symlink(timestamps_src, timestamps_target)
            
            # 映射 flow/forward_timestamps.txt
            if mode == 'train':
                forward_timestamps_src = os.path.join(subfolder_path, 'flow', 'forward_timestamps.txt')
            else:
                forward_timestamps_src = os.path.join(subfolder_path, 'flow', f'{subfolder}.csv')
            forward_timestamps_target = os.path.join(target_subfolder, 'flow', 'forward_timestamps.txt')
            # 检查文件是否存在
            if os.path.exists(forward_timestamps_src) and os.path.exists(timestamps_src):
                # 读取 image_timestamps.txt 文件并存储为列表
                with open(timestamps_src, 'r') as img_file:
                    image_timestamps = img_file.read().splitlines()

                # 准备写入 TXT 文件
                with open(forward_timestamps_target, 'w') as txt_file:
                    txt_file.write('# from_timestamp_us,to_timestamp_us,file_index\n')  # 写入表头

                    # 读取 forward_timestamps.txt 或 .csv 文件并处理
                    if mode == 'train':
                        with open(forward_timestamps_src, 'r') as flow_file:
                            flow_lines = flow_file.readlines()
                            # 从第二行开始处理 forward_timestamps.txt
                            for line in flow_lines[1:]:  # 跳过第一行
                                if line.strip():  # 确保行不为空
                                    from_timestamp, to_timestamp = map(int, line.split(',')[:2])  # 解析时间戳
                                    # 查找 from_timestamp 在 image_timestamps 中的行数
                                    try:
                                        file_index = image_timestamps.index(str(from_timestamp))
                                        # 写入 TXT
                                        txt_file.write(f'{from_timestamp},{to_timestamp},{file_index}\n')
                                    except ValueError:
                                        print(f"Warning: {from_timestamp} not found in image timestamps in {subfolder}.")
                                        continue  # 跳过这个时间戳
                    else:
                        with open(forward_timestamps_src, 'r') as flow_file:
                            csv_reader = csv.reader(flow_file)
                            next(csv_reader)  # 跳过第一行
                            for row in csv_reader:
                                if row:  # 确保行不为空
                                    from_timestamp, to_timestamp = map(int, row[:2])  # 解析时间戳
                                    # 查找 from_timestamp 在 image_timestamps 中的行数
                                    try:
                                        file_index = image_timestamps.index(str(from_timestamp))
                                        # 写入 TXT
                                        txt_file.write(f'{from_timestamp},{to_timestamp},{file_index}\n')
                                    except ValueError:
                                        print(f"Warning: {from_timestamp} not found in image timestamps in {subfolder}.")
                                        continue  # 跳过这个时间戳

            # 映射 exposure_timestamps.txt
            exposure_timestamps_src = os.path.join(subfolder_path, 'images', 'exposure_timestamps.txt')
            exposure_timestamps_target = os.path.join(target_subfolder, 'images', 'exposure_timestamps.txt')
            if os.path.exists(exposure_timestamps_src):
                os.makedirs(os.path.dirname(exposure_timestamps_target), exist_ok=True)
                os.symlink(exposure_timestamps_src, exposure_timestamps_target)

            # 映射 images
            images_rectified_src = os.path.join(subfolder_path, 'images', 'left', 'rectified')
            images_rectified_target = os.path.join(target_subfolder, 'images', 'left', 'rectified')
            if os.path.exists(images_rectified_src):
                os.makedirs(os.path.dirname(images_rectified_target), exist_ok=True)
                os.symlink(images_rectified_src, images_rectified_target)
            images_warped_src = os.path.join(subfolder_path, 'images', 'left', 'warped')
            images_warped_target = os.path.join(target_subfolder, 'images', 'left', 'ev_inf')
            if os.path.exists(images_warped_src):
                os.makedirs(os.path.dirname(images_warped_target), exist_ok=True)
                os.symlink(images_warped_src, images_warped_target)

print("映射完成！")