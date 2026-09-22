import numpy as np
import pandas as pd

from collections import deque
import matplotlib.pyplot as plt

from project_paths import CITIES_CHINA_CSV

# 假设有 cities_china.csv 文件，包含三列：city,lat,lon

# ----------- 地理位置参数 --------------------
R = 42164  # 地球静止轨道半径（公里）
r = 6371  # 地球半径 km
k = 1.380649e-23
c = 2.99792458e8  # 光速

beam_num = 100  # 波束数量
frequency_waste_threshold = 2
power_threshold = beam_num * 30
power_overload_risk_level = power_threshold / beam_num
rate_min, rate_max, rate_step = 100, 400, 100
valid_rates = np.arange(rate_min, rate_max, 100)

# lat_min, lat_max = 4.0, 54.0  # 中国纬度范围
# lon_min, lon_max = 73.0, 135.0  # 中国经度范围
# latitudes = np.linspace(lat_min, lat_max, 10)
# longitudes = np.linspace(lon_min, lon_max, 10)  # 10*10网格
# lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)
# beam_location = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
df = pd.read_csv(CITIES_CHINA_CSV)
beam_location = df[["lat", "lon"]].values
satellite_lon = 122.2  # 卫星轨道经度 角度
satellite_lat = 0.0  # GEO卫星位于赤道
satellite_lon_rad = np.deg2rad(satellite_lon)
x_sat = R * np.cos(satellite_lon_rad)
y_sat = R * np.sin(satellite_lon_rad)
z_sat = 0.0

lat_rad = np.deg2rad(beam_location[:, 0])
lon_rad = np.deg2rad(beam_location[:, 1])

x_ground = r * np.cos(lat_rad) * np.cos(lon_rad)
y_ground = r * np.cos(lat_rad) * np.sin(lon_rad)
z_ground = r * np.sin(lat_rad)

dx = x_sat - x_ground
dy = y_sat - y_ground
dz = z_sat - z_ground
distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # km


def geo_to_cartesian(lat, lon):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return np.array([x, y, z])


# -------------------- 数据速率需求参数 --------------------
rate_mean = 2e8  # 数据速率需求均值
rate_std_dev = 5e7  # 数据速率需求标准差
# -------------------- 噪声温度参数 --------------------
noise_mean = 200
noise_std_dev = 50

# -------------------- 频率参数 --------------------
f_min = 1.77e10  # Hz
f_max = 20.2e10  # Hz


class AntennaPattern:
    def transmitting_antenna_pattern(self, G_max, phi):
        """简化的发射天线方向图
        Args:
            G_max: 最大增益(dBi)
            phi: 波束偏离角(度)
        """
        beamwidth = 2 # 波束宽度0.5度

        # 主瓣区域（±0.25度）
        if abs(phi) <= beamwidth / 2:
            return G_max - 3 * (phi / (beamwidth / 2)) ** 2  # 抛物面近似

        # 旁瓣区域（>0.5度）
        else:
            # 旁瓣衰减斜率控制（确保CINR下限可达-2.35dB）
            return G_max - 30 * np.log10(abs(phi)) - 25  # 对数衰减模型

    def receiving_antenna_pattern(self, G_max, phi):
        """简化的接收天线方向图（与发射对称）"""
        return self.transmitting_antenna_pattern(G_max, phi)


class MCS:
    def __init__(self):
        self.modcod_table = [
            {"mode": "QPSK 1/4", "spectral_efficiency": 0.490243, "E_g_No": -2.35},
            {"mode": "QPSK 1/3", "spectral_efficiency": 0.656448, "E_g_No": -1.24},
            {"mode": "QPSK 2/5", "spectral_efficiency": 0.789412, "E_g_No": -0.30},
            {"mode": "QPSK 1/2", "spectral_efficiency": 0.988858, "E_g_No": 1.00},
            {"mode": "QPSK 3/5", "spectral_efficiency": 1.188304, "E_g_No": 2.23},
            {"mode": "QPSK 2/3", "spectral_efficiency": 1.322253, "E_g_No": 3.10},
            {"mode": "QPSK 3/4", "spectral_efficiency": 1.487473, "E_g_No": 4.03},
            {"mode": "QPSK 4/5", "spectral_efficiency": 1.587196, "E_g_No": 4.68},
            {"mode": "QPSK 5/6", "spectral_efficiency": 1.654663, "E_g_No": 5.18},
            {"mode": "QPSK 8/9", "spectral_efficiency": 1.766451, "E_g_No": 6.20},
            {"mode": "QPSK 9/10", "spectral_efficiency": 1.788612, "E_g_No": 6.42},
            {"mode": "8PSK 3/5", "spectral_efficiency": 1.779991, "E_g_No": 5.50},
            {"mode": "8PSK 2/3", "spectral_efficiency": 1.980636, "E_g_No": 6.62},
            {"mode": "8PSK 3/4", "spectral_efficiency": 2.228124, "E_g_No": 7.91},
            {"mode": "8PSK 5/6", "spectral_efficiency": 2.478562, "E_g_No": 9.35},
            {"mode": "8PSK 8/9", "spectral_efficiency": 2.646012, "E_g_No": 10.69},
            {"mode": "8PSK 9/10", "spectral_efficiency": 2.679207, "E_g_No": 10.98},
            {"mode": "16APSK 2/3", "spectral_efficiency": 2.637201, "E_g_No": 8.97},
            {"mode": "16APSK 3/4", "spectral_efficiency": 2.966728, "E_g_No": 10.21},
            {"mode": "16APSK 4/5", "spectral_efficiency": 3.165623, "E_g_No": 11.03},
            {"mode": "16APSK 5/6", "spectral_efficiency": 3.300184, "E_g_No": 11.61},
            {"mode": "16APSK 8/9", "spectral_efficiency": 3.523143, "E_g_No": 12.89},
            {"mode": "16APSK 9/10", "spectral_efficiency": 3.567342, "E_g_No": 13.13},
            {"mode": "32APSK 3/4", "spectral_efficiency": 3.703295, "E_g_No": 12.73},
            {"mode": "32APSK 4/5", "spectral_efficiency": 3.951571, "E_g_No": 13.64},
            {"mode": "32APSK 5/6", "spectral_efficiency": 4.119540, "E_g_No": 14.28},
            {"mode": "32APSK 8/9", "spectral_efficiency": 4.397854, "E_g_No": 15.69},
            {"mode": "32APSK 9/10", "spectral_efficiency": 4.453027, "E_g_No": 16.05}
        ]

    def get_spectral_efficiency(self, CNR):
        # 表格数据（已转换为浮点数，E_g/No单位为dB）

        # 按E_g/No升序排序
        sorted_table = sorted(self.modcod_table, key=lambda x: x["E_g_No"])

        # 遍历查找满足条件的最优模式
        max_efficiency = 0.0
        for entry in sorted_table:
            if entry["E_g_No"] <= CNR:
                if entry["spectral_efficiency"] > max_efficiency:
                    max_efficiency = entry["spectral_efficiency"]
            else:
                break  # 由于已排序，后续E_g/No更大，无需继续

        return max_efficiency if max_efficiency > 0 else sorted_table[0]["spectral_efficiency"]

    def get_Eg_No(self, spectral_efficiency):
        # 按频谱效率升序排序
        sorted_table = sorted(self.modcod_table, key=lambda x: x["E_g_No"])

        # 遍历查找满足条件的最优模式
        min_Eg_No = float('inf')  # 初始化为最大值
        for entry in sorted_table:
            if entry["spectral_efficiency"] >= spectral_efficiency:
                if entry["E_g_No"] < min_Eg_No:
                    min_Eg_No = entry["E_g_No"]
            else:
                continue  # 继续查找更大的频谱效率

        # 如果未找到满足条件的模式，返回最大频谱效率对应的 E_g/No
        if min_Eg_No == float('inf'):
            return sorted_table[-1]["E_g_No"]
        else:
            return min_Eg_No


# -------------------- 环境组件 --------------------
class SatelliteFreqState:
    def __init__(self):
        self.freq_pool = np.zeros((8, 100))  # 8行100列的频时矩阵

    def update_state(self, group, freq, slots):
        """
        更新资源池状态
        :param group: 波束组 (0-7)
        :param freq: 起始频率位置 (0-99)
        :param slots: 分配的频率槽数量
        """
        if group < 0 or group >= 8:
            raise ValueError("group must be between 0 and 7")
        if freq < 0 or freq >= 100:
            raise ValueError("freq must be between 0 and 99")
        if slots < 0 or slots > 10:
            raise ValueError("slots must be non-negative and slots must be > 10")

        # 更新频时矩阵，将分配的时隙标记为1
        self.freq_pool[group, freq:freq + slots] += 1

    def get_state(self):
        """基础状态表示"""
        return self.freq_pool.flatten()

    def reset_state(self):
        self.freq_pool = np.zeros((8, 100))
        return self.freq_pool.flatten()


class BeamInfo:
    def __init__(self, shuffle=True):
        np.random.seed(None)
        self.beam_location = beam_location
        self.beam_G_Tx_max = np.ones((beam_num, 1)) * 50  # 发射天线主瓣增益
        self.UE_G_Rx_max = np.ones((beam_num, 1)) * 40
        self.distances = distances  # km
        rate_noise = np.random.normal(loc=0, scale=np.sqrt(400), size=(beam_num, 1))
        rate_noise = np.round(rate_noise).astype(int)
        self.rate = (np.random.choice(valid_rates, size=(beam_num, 1)) + rate_noise) * 1e6
        self.rate = np.where(self.rate < 0, 1e8, self.rate)
        self.noise_temperature = np.ones((beam_num, 1)) * 290
        self.beam_info = {
            'beam_location': self.beam_location,
            'beam_G_Tx_max': self.beam_G_Tx_max,
            'UE_G_Rx_max': self.UE_G_Rx_max,
            'distances': self.distances,
            'rate': self.rate,
            'noise_temperature': self.noise_temperature
        }
        self.shuffle_indices = np.arange(beam_num)
        if shuffle:
            self.shuffle()

        # 应用初始乱序
        self.apply_shuffle()

    def shuffle(self):
        """生成新的随机排列"""
        np.random.shuffle(self.shuffle_indices)

    def apply_shuffle(self):
        """应用当前乱序到所有波束属性"""
        self.beam_location = self.beam_location[self.shuffle_indices]
        self.beam_G_Tx_max = self.beam_G_Tx_max[self.shuffle_indices]
        self.UE_G_Rx_max = self.UE_G_Rx_max[self.shuffle_indices]
        self.distances = self.distances[self.shuffle_indices]
        self.rate = self.rate[self.shuffle_indices]
        self.noise_temperature = self.noise_temperature[self.shuffle_indices]

        self.beam_info = {
            'beam_location': self.beam_location,
            'beam_G_Tx_max': self.beam_G_Tx_max,
            'UE_G_Rx_max': self.UE_G_Rx_max,
            'distances': self.distances,
            'rate': self.rate,
            'noise_temperature': self.noise_temperature
        }


class CurrentBeamState:
    def __init__(self, beam_info):
        self.beam_id = 0
        self.history_beam = []
        self.beam_info = beam_info

    def reset_state(self):
        """返回当前波束的状态向量（固定3维）"""

        self.beam_state = {
            # 确保为标量值（关键修改）
            'required_rate': self.beam_info['rate'][self.beam_id].item(),  # 从数组中提取标量
            'beam_location': self.beam_info['beam_location'][self.beam_id].tolist()  # 转换为列表
        }
        rate = self.beam_state['required_rate']
        lat, lon = self.beam_state['beam_location']

        # ---------- 归一化开始 ----------
        # 1. 标准化数据速率需求
        rate_mean = (rate_min + rate_max - rate_step) / 2 * 1e6
        normalized_rate = rate / rate_mean

        # 2. 归一化地理位置
        # 纬度归一化到 [-1, 1]
        lat_min, lat_max = 4.0, 54.0
        normalized_lat = 2 * (lat - (lat_max + lat_min) / 2) / (lat_max - lat_min)

        # 经度归一化到 [-1, 1]
        lon_min, lon_max = 73.0, 135.0
        normalized_lon = 2 * (lon - (lon_max + lon_min) / 2) / (lon_max - lon_min)
        current_beam_state = np.array([
            normalized_rate,
            normalized_lat,
            normalized_lon
        ], dtype=np.float32)
        # ---------- 归一化结束 ----------
        hist_deque = deque(self.history_beam)
        hist_deque.appendleft(current_beam_state)
        self.history_beam = list(hist_deque)
        self.beam_id += 1
        # print(self.history_beam)
        return self.history_beam

    def update_state(self, action):
        """返回当前波束的状态向量（固定3维）"""

        self.beam_state = {
            # 确保为标量值（关键修改）
            'required_rate': self.beam_info['rate'][self.beam_id].item(),  # 从数组中提取标量
            'beam_location': self.beam_info['beam_location'][self.beam_id].tolist()  # 转换为列表
        }
        rate = self.beam_state['required_rate']
        lat, lon = self.beam_state['beam_location']

        # ---------- 归一化开始 ----------
        # 1. 标准化数据速率需求
        rate_mean = (rate_min + rate_max - rate_step) / 2 * 1e6
        normalized_rate = rate / rate_mean

        # 2. 归一化地理位置
        # 纬度归一化到 [-1, 1]
        lat_min, lat_max = 4.0, 54.0
        normalized_lat = 2 * (lat - (lat_max + lat_min) / 2) / (lat_max - lat_min)

        # 经度归一化到 [-1, 1]
        lon_min, lon_max = 73.0, 135.0
        normalized_lon = 2 * (lon - (lon_max + lon_min) / 2) / (lon_max - lon_min)
        current_beam_state = np.array([
            normalized_rate,
            normalized_lat,
            normalized_lon,
            action[0] / 8,
            action[1] / 100,
            action[2] / 10,
            action[3] / 50
        ], dtype=np.float32)
        # ---------- 归一化结束 ----------
        hist_deque = deque(self.history_beam)
        hist_deque.appendleft(current_beam_state)
        self.history_beam = list(hist_deque)
        self.beam_id += 1
        # print(self.history_beam)
        return self.history_beam

    def get_state(self):
        history_beam_state = np.array(
            ([item for sublist in self.history_beam for item in sublist] + [0.0] * (beam_num - self.beam_id) * 7),
            dtype=np.float32)
        return history_beam_state

    def reset(self, beam_info):
        self.beam_id = 0
        self.history_beam = []
        self.beam_info = beam_info


# -------------------- 卫星环境类 --------------------
class Env:
    def __init__(self, satellite_freq_state, beam_info, hist_beam_state):
        self.beam_idx = 0  # 当前波束
        self.beam_info = beam_info.beam_info
        self.beam_num = beam_num
        self.satellite_freq_state = satellite_freq_state
        self.current_beam_state = hist_beam_state
        self.antenna_pattern = AntennaPattern()
        self.mcs = MCS()
        self.history_beam = []  # 记录当前回合已分配波束的动作
        self.history_cinr = []  # 记录每一个已分配频率槽CINR
        self.total_power = 0
        self.total_satisfaction = 0
        self.overload = False
        self.beta = sum(self.beam_info['rate']) / self.beam_num
        self.alpha = power_threshold / self.beam_num
        self.beam_bps = []  # 记录每个波束的数据速率
        self.occupy = 0
        self.interfere_num = 0
        self.interfere_in_1000km_num = 0
    def get_combined_state(self, beam_state):
        """
        将 CurrentBeamState 和 SatelliteFreqState 的状态拼接为一个向量
        :param beam_state: CurrentBeamState 的实例
        :param freq_state: SatelliteFreqState 的实例
        :return: 拼接后的状态向量
        """
        beam_state_vector = beam_state.get_state()

        return beam_state_vector

    def reset(self, beam_info):
        self.beam_idx = 0
        self.occupy = 0
        self.interfere_num = 0
        self.interfere_in_1000km_num = 0
        self.satellite_freq_state.reset_state()
        self.history_beam = []
        self.history_cinr = []
        self.beam_bps = []  # 记录每个波束的数据速率
        self.total_power = 0
        self.overload = False
        self.total_satisfaction = 0
        self.beam_info = beam_info.beam_info
        self.current_beam_state.reset(self.beam_info)
        self.current_beam_state.reset_state()
        state = self.get_combined_state(self.current_beam_state)
        return state

    def calculate_beam_angle(self, beam_lat, beam_lon, current_lat, current_lon):
        """
        计算波束主瓣指向点与当前点相对于卫星的夹角
        输入：
          beam_lat, beam_lon : 波束主瓣指向的纬度(°)、经度(°)
          current_lat, current_lon : 当前点的纬度(°)、经度(°)
        输出：
          角度（度）
        """
        # 卫星轨道参数（GEO, 东经122.2度）

        # 卫星坐标（东经122.2度，赤道）
        sat_lon_rad = np.deg2rad(satellite_lon)
        x_sat = R * np.cos(sat_lon_rad)
        y_sat = R * np.sin(sat_lon_rad)
        z_sat = 0.0

        # 计算各点坐标
        beam_point = geo_to_cartesian(beam_lat, beam_lon)
        current_point = geo_to_cartesian(current_lat, current_lon)

        # 计算卫星到各点的向量
        vec_beam = beam_point - [x_sat, y_sat, z_sat]
        vec_current = current_point - [x_sat, y_sat, z_sat]

        # 计算向量夹角
        cos_theta = np.dot(vec_beam, vec_current) / (np.linalg.norm(vec_beam) * np.linalg.norm(vec_current))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 处理浮点误差
        return np.rad2deg(np.arccos(cos_theta))

    def haversine(self, beam_lat, beam_lon, interfere_lat, interfere_lon):
        """
        向量化球面距离计算（支持标量/数组输入）

        参数:
        lat1, lon1 : 数组或标量 - 第一组坐标的纬度和经度（十进制度）
        lat2, lon2 : 数组或标量 - 第二组坐标的纬度和经度（十进制度）

        返回:
        距离数组（单位：米），形状由输入广播规则决定
        """
        # 转换为弧度
        beam_lat = np.radians(beam_lat)
        beam_lon = np.radians(beam_lon)
        interfere_lat = np.radians(interfere_lat)
        interfere_lon = np.radians(interfere_lon)

        # 计算差值
        dlat = interfere_lat - beam_lat
        dlon = interfere_lon - beam_lon

        # Haversine公式向量化计算
        a = np.sin(dlat / 2) ** 2 + np.cos(beam_lat) * np.cos(interfere_lat) * np.sin(dlon / 2) ** 2
        d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return 6371 * d

    def receiving_power(self, action):
        group, freq, slots, power = action
        # 获取当前波束的地理参数
        # beam_location = self.current_beam_state.beam_state['beam_location']
        f = 1.77e10 + (freq + slots / 2) * 2.5e7  # 当前波束频率中心点

        # 天线参数（波束已对准地面站，phi=0）
        phi_tx = 0.0  # 波束已对准，偏离角为0度
        phi_rx = 0.0

        # 直接获取最大天线增益（phi=0时的增益）
        G_tx_max = self.beam_info['beam_G_Tx_max'][self.beam_idx][0]  # 发射增益最大值
        G_rx_max = self.beam_info['UE_G_Rx_max'][self.beam_idx][0]  # 接收增益最大值
        G_tx = self.antenna_pattern.transmitting_antenna_pattern(G_tx_max, phi_tx)
        G_rx = self.antenna_pattern.receiving_antenna_pattern(G_rx_max, phi_rx)

        # 计算路径损耗（转换为米）
        distance = self.beam_info['distances'][self.beam_idx] * 1000
        wavelength = c / f
        path_loss = (4 * np.pi * distance / wavelength) ** 2
        path_loss_db = 10 * np.log10(path_loss)

        # 计算接收功率（dB）
        P_rx = power + G_tx + G_rx - path_loss_db
        return P_rx

    def interfere_beams(self, action):
        """
        检测与当前波束发生频率冲突的历史波束，并返回冲突详情
        :return: 冲突列表 [ (冲突波束ID, [冲突频率槽索引]), ... ]
        """
        # 解析当前波束参数
        # (f"in{action}")
        # print(f"hisbeam{self.history_beam}")
        group, freq, slots, _ = action
        polarization = group % 2  # 0,1为不同极化方式
        # current_beam_id = self.beam_idx
        freq_start = freq
        freq_end = freq + slots - 1  # 频率槽闭区间[freq_start, freq_end]

        conflict_list = []
        interfere_in_1000km = False
        interfere_all = 0

        # 遍历历史波束记录
        for hist_entry in self.history_beam:
            hist_beam_id, hist_group, hist_freq, hist_slots, _ = hist_entry
            hist_polarization = hist_group % 2
            # 排除自身
            if hist_polarization != polarization or (hist_group == group and hist_freq == freq):
                # print("sss")
                continue
            # 计算历史波束频率范围
            hist_freq_start = hist_freq
            hist_freq_end = hist_freq + hist_slots - 1

            # 计算重叠区域
            overlap_start = max(freq_start, hist_freq_start)
            overlap_end = min(freq_end, hist_freq_end)

            # 检测冲突
            if overlap_start <= overlap_end:
                # 生成冲突频率槽索引列表
                beam_lat, beam_lon = self.beam_info['beam_location'][self.beam_idx]
                interfere_lat, interfere_lon = self.beam_info['beam_location'][hist_beam_id]
                distance = self.haversine(beam_lat, beam_lon, interfere_lat, interfere_lon)
                conflict_slots = list(range(overlap_start, overlap_end + 1))
                if distance <= 1000:
                    interfere_in_1000km = True
                conflict_group = hist_group
                # 添加到冲突列表
                conflict_list.append((hist_beam_id, conflict_group, conflict_slots))

        # print(conflict_list)
        return conflict_list, interfere_in_1000km

    def interfere_power(self, conflict_list, action):
        # print(f"conflict_list{conflict_list}")
        beam_lat, beam_lon = self.beam_info['beam_location'][self.beam_idx]
        group, freq, slots, power = action
        interfere_power_info = []
        for conflict_slots_info in conflict_list:
            conflict_beam, conflict_group, conflict_slots = conflict_slots_info
            for conflict_slot in conflict_slots:
                conflict_lat, conflict_lon = self.beam_info['beam_location'][conflict_beam]
                phi = self.calculate_beam_angle(beam_lat, beam_lon, conflict_lat, conflict_lon)
                P_conflict = self.history_beam[conflict_beam][4]
                G_conflict_tx_max = self.beam_info['beam_G_Tx_max'][conflict_beam]
                G_conflict_tx = self.antenna_pattern.transmitting_antenna_pattern(G_conflict_tx_max, phi)
                G_conflict_rx_max = self.beam_info['UE_G_Rx_max'][conflict_beam]
                G_conflict_rx = self.antenna_pattern.receiving_antenna_pattern(G_conflict_rx_max, 0)
                conflict_distance = self.beam_info['distances'][self.beam_idx] * 1000

                conflict_path_loss = (4 * np.pi * conflict_distance / (
                        c / (1.77e10 + 2.5e7 * (conflict_slot + 0.5)))) ** 2
                conflict_path_loss_db = 10 * np.log10(conflict_path_loss)
                conflict_rx_power_db = P_conflict + G_conflict_tx + G_conflict_rx - conflict_path_loss_db  # 对当前波束产生的干扰功率
                # print(f"P_conflict:{P_conflict},G_conflict_tx{G_conflict_tx},G_conflict_rx{G_conflict_rx},conflict_path_loss_db{conflict_path_loss_db}")

                G_tx_max = self.beam_info['beam_G_Tx_max'][self.beam_idx][0]  # 发射增益最大值
                G_rx_max = self.beam_info['UE_G_Rx_max'][self.beam_idx][0]  # 接收增益最大值
                G_to_conflict_tx = self.antenna_pattern.transmitting_antenna_pattern(G_tx_max, phi)
                G_to_conflict_rx = self.antenna_pattern.receiving_antenna_pattern(G_rx_max, 0)
                to_conflict_distance = self.beam_info['distances'][conflict_beam] * 1000

                to_conflict_path_loss = (4 * np.pi * to_conflict_distance / (
                        c / (1.77e10 + 2.5e7 * (conflict_slot + 0.5)))) ** 2
                to_conflict_path_loss_db = 10 * np.log10(to_conflict_path_loss)
                to_conflict_rx_power_db = power + G_to_conflict_tx + G_to_conflict_rx - to_conflict_path_loss_db  # 当前波束对干扰波束造成的干扰

                interfere_power_info.append(
                    [self.beam_idx, group, conflict_rx_power_db.item(), conflict_beam, conflict_group,
                     to_conflict_rx_power_db, conflict_slot])
        interfere_flag = False
        if len(interfere_power_info) > 0:
            interfere_flag = True
        return interfere_power_info, interfere_flag

    def current_rate_calculate(self, P_rx, interfere_power_info, action):  # 计算当前波束数据速率
        group, freq, slots, power = action
        current_slots = range(freq, freq + slots)
        noise_temp = self.beam_info['noise_temperature'][self.beam_idx]
        P_rx_linear = 10 ** (P_rx / 10)
        data_rates = []
        interfere_noise_linear = []
        efficiency_avg = 0
        for slot in current_slots:
            # 收集该槽的所有干扰功率（dBW）
            interference_dB = [entry[2] for entry in interfere_power_info if entry[6] == slot]
            # 转换干扰功率为线性值并求和（W）
            interference_linear = sum(10 ** (power_db / 10) for power_db in interference_dB)
            # print(f"P_rx_dB{P_rx}")
            # print(f"interference_dB{interference_dB}")
            noise_linear = k * noise_temp * 2.5e7
            # print(f"N_dB{10*np.log10(noise_linear)}")
            # print(f"noise_linear{noise_linear}")
            cinr_linear = P_rx_linear / (noise_linear + interference_linear)
            interfere_noise_linear.append(noise_linear + interference_linear)  # TODO:头昏脑胀
            cinr_db = 10 * np.log10(cinr_linear.item()) - 5  # 5dB其他损耗
            # print(cinr_db)
            self.history_cinr.append([group, slot, cinr_db, P_rx, self.beam_idx])  # 记录历史cinr
            spectral_efficiency = self.mcs.get_spectral_efficiency(cinr_db)
            efficiency_avg += spectral_efficiency
            # 计算数据速率 = 频谱效率 × 带宽
            data_rate = spectral_efficiency * 2.5e7
            data_rates.append(data_rate)
        if len(current_slots) > 0:
            efficiency_avg = efficiency_avg / len(current_slots)
        else:
            efficiency_avg = 0
        interfere_noise_linear_value = sum(
            interfere_noise_linear) if interfere_noise_linear != [] else k * noise_temp * 2.5e7
        return sum(data_rates), self.beam_info['rate'][self.beam_idx], efficiency_avg, interfere_noise_linear_value

    def interfere_effect(self, interfere_power_info):
        for entry in interfere_power_info:
            current_beam, current_group, current_interfere_power_db, conflict_beam, conflict_group, to_conflict_rx_power_db, conflict_slot = entry
            # 当前频率槽原CINR
            # print(f"hist{self.history_cinr}")
            i = 0  # 记录history_cinr循环到了哪一行
            for cinr_info in self.history_cinr:
                search_group, search_slot, origin_cinr_db, search_P_rx, search_idx = cinr_info
                # print(search_P_rx)
                if not (search_group == conflict_group and search_slot == conflict_slot):
                    i += 1
                    continue
                origin_spectral_efficiency = self.mcs.get_spectral_efficiency(origin_cinr_db)
                origin_data_rate = origin_spectral_efficiency * 2.5e7
                origin_cinr = 10 ** (origin_cinr_db / 10)  # 原cinr转换为线性值
                conflict_beam_power = 10 ** (search_P_rx / 10)
                # print(conflict_beam_power)
                conflict_cinr = conflict_beam_power / (
                        (1 / origin_cinr) * conflict_beam_power + 10 ** (to_conflict_rx_power_db / 10))
                conflict_cinr_db = 10 * np.log10(conflict_cinr)
                # print(f"origin{origin_cinr}conflict{conflict_cinr}")
                self.history_cinr[i][2] = conflict_cinr_db
                conflict_spectral_efficiency = self.mcs.get_spectral_efficiency(conflict_cinr_db)
                conflict_data_rate = conflict_spectral_efficiency * 2.5e7
                rate_diff = (origin_data_rate - conflict_data_rate)
                self.beam_bps[search_idx] -= rate_diff
                i += 1

    def calculate_total_satisfaction(self):
        required_rate = np.array(self.beam_info['rate'][0:self.beam_idx + 1])
        allocate_rate = np.array(self.beam_bps)
        total_satisfaction = np.minimum(allocate_rate / required_rate.flatten(), 1)
        if np.any(allocate_rate < 0):
            input()
        return total_satisfaction.sum()

    def get_first_free_slot(self, action):
        freq_group, _ = action
        group = self.beam_idx % 8
        search_line = self.satellite_freq_state.freq_pool[group].reshape(10, 10)[freq_group]
        in_group_idx = np.where(search_line == 0)[0]
        if len(in_group_idx) > 0:
            in_group_idx = in_group_idx[0]
        else:
            in_group_idx = 0
        out_group_idx = freq_group * 10 + in_group_idx
        return out_group_idx

    # def get_power_budget_constraint(self):
    #     remaining_beams = self.beam_num - self.beam_idx
    #     remaining_power = max(0, power_threshold - self.total_power)
    #
    #     # 基础预算 + 安全余量
    #     return remaining_power / remaining_beams

    def SGM_rate(self, allocate_rate, required_rate):
        SI = allocate_rate / required_rate
        assert SI >= 0, "Invalid SI"
        delta_S_I = allocate_rate - required_rate
        if SI == 0:
            sgm = np.array([0])
            # print(sgm)
        else:
            labda = (SI - 1) + delta_S_I / self.beta * 1j if SI >= 1 else (1 - 1 / SI) + delta_S_I / self.beta * 1j
            sgm = 1 - (1 - np.exp(-(abs(labda)))) ** 3
        # if abs(delta_S_I) > 5e7:
        #     sgm = sgm - abs(abs(delta_S_I) - 5e7) / 1e8

        return sgm

    def SGM_power(self, allocate_power, ideal_power):
        SI = allocate_power / ideal_power
        assert SI >= 0, "Invalid SI"
        delta_S_I = allocate_power - ideal_power
        if SI == 0:
            sgm = 0
        else:
            labda = (SI - 1) + delta_S_I / self.alpha * 1j if SI >= 1 else (1 - 1 / SI) + delta_S_I / self.alpha * 1j
            sgm = 1 - (1 - np.exp(-(abs(labda)))) ** 3
        if abs(delta_S_I) > 5e7:
            sgm = sgm - abs(abs(delta_S_I) - 5e7) / 1e8
        return sgm

    def get_required_power(self, required_rate, slots, P_rx, power, interfere_noise_db):
        required_efficiency = required_rate / (slots * 2.5e7)
        ideal_cinr = self.mcs.get_Eg_No(required_efficiency)
        power_db = 10 * np.log10(power)
        ideal_power_db = ideal_cinr + interfere_noise_db - (P_rx - power_db)
        ideal_power = 10 ** (ideal_power_db / 10)
        power_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        # 在 power_levels 中向上取整
        rounded_ideal_power_db = min((level for level in power_levels if level >= ideal_power),
                                     default=power_levels[-1])

        return rounded_ideal_power_db

    def band_utilization(self, slots, util_factor=10):
        util = slots / 800 * util_factor
        return util

    def get_occupy(self):
        return self.occupy
    
    def get_action_mask(self):
        freq_pool = self.satellite_freq_state.get_state().reshape(8,100)[self.beam_idx % 8].reshape(10,10)
        counts = freq_pool.sum(axis=1)
        mask = (counts < 8).astype(int)  # True->1, False->0
        if mask.sum() == 0:
            mask += 1
        return np.array(mask)

    def step(self, action):
        """执行动作并返回环境状态
        Args:
            action: Tuple (group, freq, slots, power)
                group: 频率组 (0-7)
                freq: 起始频点 (0-99)
                slots: 频率槽数量 (1-100)
                power: 发射功率 (dBW)
        Returns:
            next_state: 新状态向量
            reward: 奖励值
            done: 是否完成所有波束分配
            info: 调试信息字典
        """
        # -------------------- 1. 参数解析与校验 --------------------
        done = (self.beam_idx + 1 > self.beam_num - 1)
        out = 0  # 出界数
        if action[2] + action[1] > 100:  # 出界数
            out = action[2] + action[1] - 100
            # print(out)
            action[2] = action[2] - out

        slots_range = range(action[1], action[1] + action[2])
        no_occupy = 0
        origin_slots = action[2]
        for slot in slots_range:
            if self.satellite_freq_state.freq_pool[action[0]][slot] > 0:
                action[2] = no_occupy
                break
            no_occupy += 1
        occupy = origin_slots - no_occupy  # 被占用的槽数
        self.occupy += occupy
        group, freq, slots, power_dB = action
        power = 10 ** (power_dB / 10)
        power_exceed_penalty = 0
        self.total_power += power

        # excess_ratio = 0
        # power_budget = self.get_power_budget_constraint()
        # if power_budget > 0:
        #     if power > power_budget:
        #         excess_ratio = (power - power_budget) / power_budget
        # else:
        #     excess_ratio = 1

        if self.total_power > power_threshold and not self.overload:
            print("overload")
            self.overload = True
            power_exceed_penalty = 100

        # -------------------- 2. 更新系统状态 --------------------
        if not done:
            self.satellite_freq_state.update_state(group, freq, slots)
            self.current_beam_state.update_state(action)
        # print(f"here{self.satellite_freq_state.freq_pool}")
        self.history_beam.append((self.beam_idx, group, freq, slots, power_dB))

        # -------------------- 3. 计算当前波束性能 --------------------
        # 接收功率计算
        P_rx = self.receiving_power(action)

        # 干扰检测与功率计算
        conflict_list, interfere_in_1000km = self.interfere_beams(action)
        # adjacent_penalty = interfere_in_1000km
        # print(adjacent_penalty)
        interfere_power_info, interfere_flag = self.interfere_power(conflict_list, action)
        interfere_penalty = 0
        # print(f"interfere_power_info{interfere_power_info}")
        if interfere_flag:
            self.interfere_num += 1
            if interfere_in_1000km:
                self.interfere_in_1000km_num += 1
                interfere_penalty = 10
                #print(10)
            else:
                interfere_penalty = 1
                #print(1)
        # print(f"interfere_power_info{interfere_power_info}")

        # 当前波束数据速率满足率
        allocate_rate, required_rate, efficiency_avg, interfere_noise_linear = self.current_rate_calculate(P_rx,
                                                                                                           interfere_power_info,
                                                                                                           action)
        self.beam_bps.append(allocate_rate)
        self.interfere_effect(interfere_power_info)

        interfere_noise_db = 10 * np.log10(interfere_noise_linear)  # TODO:
        # ideal_power = self.get_required_power(
        #     self.beam_info['rate'][self.beam_idx],
        #     origin_slots,
        #     P_rx,
        #     power,
        #     interfere_noise_db
        # )
        current_satisfaction = allocate_rate / required_rate
        current_satisfaction = current_satisfaction.item()
        satisfaction_superfluity = 0

        if current_satisfaction > 1:
            current_satisfaction = 1

        sgm_rate = self.SGM_rate(allocate_rate, required_rate)
        # sgm_power = self.SGM_power(power, ideal_power)
        band_util = self.band_utilization(slots)

        # 被干扰波束的满足率损失
        satisfaction_increment = self.calculate_total_satisfaction() - self.total_satisfaction
        if satisfaction_increment >= 1:
            sat = 1
        else:
            sat = 0
        # -------------------- 5. 奖励函数设计 --------------------
        self.total_satisfaction = self.calculate_total_satisfaction()
        out_penalty = out
        occupy_penalty = occupy

        # 总奖励
        reward = (sat
                  + sgm_rate
                  #+ band_util
                  #+ satisfaction_increment
                #   - out_penalty
                #   - occupy_penalty
                #   - interfere_penalty
                #   #   current_satisfaction
                #   #   - satisfaction_loss
                #   #   - out #* 2.5e7 * efficiency_avg / self.beam_info['rate'][self.beam_idx]
                #   #   - occupy #* 2.5e7 * efficiency_avg / self.beam_info['rate'][self.beam_idx]
                #   #   - adjacent_penalty# * 2.5e7 * efficiency_avg / self.beam_info['rate'][self.beam_idx]
                #   #   # - satisfaction_superfluity
                   - power_exceed_penalty
                  # - power_overload_level/100
                  # + efficiency_avg
                  # - excess_ratio
                  )
        # print(power_overload_level)
        # print(f"base_reward{base_reward},penalty{penalty},out{out},occupy{occupy}")
        next_state = self.get_combined_state(self.current_beam_state)

        info = {
            "beam_id": self.beam_idx,
            "current_satisfaction": current_satisfaction,
            "frequency_utilization": np.mean(self.satellite_freq_state.freq_pool),
            "total_satisfaction": self.total_satisfaction,
            "satisfaction_increment": satisfaction_increment,
            "success_slots": slots,
            "spectral_efficiency_avg": efficiency_avg,
            "power": power,
            "is_interfere": interfere_flag,
            "out": out,
            "occupy": occupy,
            "required_rate": self.beam_info['rate'][self.beam_idx],
            "location": self.beam_info['beam_location'][self.beam_idx],
            "group": group,
            "freq": freq,
            "origin_slots": origin_slots,
            "sgm_rate": sgm_rate,
            #"sgm_power": sgm_power,
            "interfere_in_1000km": interfere_in_1000km
        }
        # -------------------- 6. 状态转移与终止判断 --------------------
        self.beam_idx += 1

        # print(next_state)
        # -------------------- 7. 构建返回信息 --------------------

        # print(action)

        # print(f"波束{self.beam_idx}动作: {next_state}")
        # print(f"干扰功率条目数: {len(interfere_power_info)}")
        # print(f"当前满足率: {current_satisfaction}")
        # print(f"当前reward{reward}")
        # print(f"Action: G{action[0]} F{action[1]} S{action[2]} P{action[3]}")
        #
        # print(f"Reward components: {current_satisfaction}|{satisfaction_loss}|{out}|{occupy}|{adjacent_penalty}")
        return next_state, float(reward), done, info


def plot_antenna_pattern():
    G_max = 40
    angles = np.linspace(-2, 2, 1000)

    antenna = AntennaPattern()
    gains = np.array([antenna.transmitting_antenna_pattern(G_max, phi) for phi in angles])

    main_lobe_indices = np.where(np.abs(angles) <= 0.25)[0]

    plt.figure(figsize=(12, 6))

    # 主瓣
    plt.plot(angles[main_lobe_indices], gains[main_lobe_indices],
             'r', lw=3, label='Main Lobe')

    # 旁瓣
    plt.plot(np.delete(angles, main_lobe_indices),
             np.delete(gains, main_lobe_indices),
             'b--', lw=2, label='Side Lobe')

    plt.scatter([-0.25, 0.25], [G_max - 3, G_max - 3],
                c='green', label='3dB Beamwidth')
    plt.axvline(0, color='k', linestyle=':', alpha=0.5)

    plt.title("Antenna Radiation Pattern")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Gain (dBi)")
    plt.xticks(np.arange(-2, 2.5, 0.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(-10, 45)
    plt.tight_layout()
    filepath = "antenna_pattern.png"
    plt.savefig(filepath, dpi=300)
    plt.show()

# plot_antenna_pattern()
