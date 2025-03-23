import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot
import cv2

class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        # returns (x, y, z, w)
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler('xyz', degrees=True)

    def __matmul__(self, other):
        # a (self), b (other)
        # returns a @ b
        #
        # R_A | t_A   R_B | t_B   R_A @ R_B | R_A @ t_B + t_A
        # --------- @ --------- = ---------------------------
        # 0   | 1     0   | 1     0         | 1
        #
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        #           R_AB  | A_t_AB
        # T_AB =    ------|-------
        #           0     | 1
        #
        # to be converted to
        #
        #           R_BA  | B_t_BA    R_AB.T | -R_AB.T @ A_t_AB
        # T_BA =    ------|------- =  -------|-----------------
        #           0     | 1         0      | 1
        #
        # This is numerically more stable than matrix inversion of T_AB
        rotation = self._rotation.inv()
        translation = - rotation.apply(self._translation)
        return Transform(translation, rotation)

def frame2event(confpath, seg_label):
    # Get mapping for this sequence:
    assert confpath.exists()
    conf = OmegaConf.load(confpath)

    K_r0 = np.eye(3)
    K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']
    K_r1 = np.eye(3)
    K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']

    R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
    R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))

    T_r0_0 = Transform.from_rotation(R_r0_0)
    T_r1_1 = Transform.from_rotation(R_r1_1)
    T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))

    T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
    R_r1_r0_matrix = T_r1_r0.R().as_matrix()
    P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)

    ht = 480
    wd = 640
    # coords: ht, wd, 2
    coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
    # coords_hom: ht, wd, 3
    coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
    # mapping: ht, wd, 3
    mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
    # mapping: ht, wd, 2
    mapping = (mapping/mapping[..., -1][..., None])[..., :2]
    mapping = mapping.astype('float32')

    label_out = cv2.remap(seg_label, mapping, None, interpolation=cv2.INTER_CUBIC)
    # label_out = label_out[:-40, :, :]
    # assert label_out.shape == (440, 640, 3)
    # # 保存rgb格式的label_out
    # # 将 numpy 数组转换为 PIL 图像
    # label_out_ = Image.fromarray(label_out, 'RGB')

    # # 保存图像
    # label_out_.save(save_path)
    return label_out

class Sequence:
    def __init__(self, dataset_dir: Path, seq_name: str):
        self.dataset_dir = dataset_dir
        self.seq_name = seq_name
        self.img_dir = dataset_dir / "images" / "left" / "rectified"
        self.timestamps_file = dataset_dir / "images" / "timestamps.txt"
        self.flow_timestamps_file = dataset_dir / "flow" / "forward_timestamps.txt"
        self.save_img_cur_dir = dataset_dir / "images" / "left" / "warped"
        self.save_img_cur_dir.mkdir(parents=True, exist_ok=True)
        self.confpath = dataset_dir / 'calibration' / 'cam_to_cam.yaml'

    def get_img(self, filepath: Path):
        assert filepath.is_file()
        img = Image.open(str(filepath))
        img = np.array(img)
        return img

    def get_img_warp(self, img_path: Path):
        img = self.get_img(img_path)
        img_warp = frame2event(self.confpath, img)
        assert img_warp.shape == (480, 640, 3)
        return img_warp

    def process_images(self):
        if not os.path.exists(self.flow_timestamps_file):
            return
        # 读取 image_timestamps.txt 文件并存储为列表
        with open(self.timestamps_file, 'r') as img_file:
            image_timestamps = img_file.read().splitlines()

        # 读取 forward_timestamps.txt 文件并处理
        with open(self.flow_timestamps_file, 'r') as flow_file:
            flow_lines = flow_file.readlines()

            # 从第二行开始处理 forward_timestamps.txt
            for line in tqdm(flow_lines[1:]):  # 跳过第一行
                if line.strip():  # 确保行不为空
                    from_timestamp, to_timestamp = map(int, line.split(',')[:2])  # 解析时间戳
                    # 查找 from_timestamp 在 image_timestamps 中的行数
                    try:
                        file_index = image_timestamps.index(str(from_timestamp))
                    except ValueError:
                        print(f"Warning: {from_timestamp} not found in image timestamps in {subfolder}.")

                img_filename_current = self.img_dir / f"{str(file_index).zfill(6)}.png"
                img_filename_100ms = self.img_dir / f"{str(file_index+2).zfill(6)}.png"

                for img_filename in [img_filename_current, img_filename_100ms]:
                    # if os.file exist self.save_img_cur_dir / img_filename.name continue
                    if (self.save_img_cur_dir / img_filename.name).is_file():
                        continue
                    if img_filename.is_file():
                        img_warp = self.get_img_warp(img_filename)
                        img_warp = Image.fromarray(img_warp.astype(np.uint8))
                        img_warp.save(self.save_img_cur_dir / img_filename.name)

if __name__ == "__main__":
    dataset_dir = Path('./DSEC/train')

    # 遍历源目录中的所有子文件夹
    for subfolder in sorted(os.listdir(dataset_dir)):
        subfolder_path = os.path.join(dataset_dir, subfolder)

        # 确保是文件夹
        if os.path.isdir(subfolder_path):
            seq_name = subfolder  # Replace with the actual sequence name
            print(f"Processing sequence: {seq_name}")
            seq = Sequence(Path(subfolder_path), seq_name)
            seq.process_images()