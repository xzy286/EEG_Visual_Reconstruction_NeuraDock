import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
import seaborn as sns
import os
def remove_large_outliers(eeg_data, fs, threshold=100):
    """
    去除大于指定阈值的异常值，并打印异常发生的时间和持续时长。
    对于连续的异常片段，使用线性插值进行修复。
    
    参数:
    eeg_data: np.array, 形状 (channels, points)
    fs: int, 采样率 (用于计算时间)
    threshold: float, 阈值，默认 1000
    
    返回:
    cleaned_data: 清洗后的数据
    """
    cleaned_data = eeg_data.copy()
    n_channels, n_points = cleaned_data.shape
    
    print(f"\n=== 异常值检测报告 (阈值 > {threshold}) ===")
    
    total_outlier_events = 0
    
    for ch in range(n_channels):
        # 1. 找到该通道所有异常值的索引
        outlier_indices = np.where(np.abs(cleaned_data[ch, :]) > threshold)[0]
        
        if len(outlier_indices) == 0:
            continue
            
        # 2. 将连续的索引分组 (例如 [10, 11, 12, 50, 51] -> [[10, 11, 12], [50, 51]])
        # np.diff(indices) > 1 的位置即为断点
        split_locs = np.where(np.diff(outlier_indices) > 1)[0] + 1
        outlier_groups = np.split(outlier_indices, split_locs)
        
        for group in outlier_groups:
            total_outlier_events += 1
            start_idx = group[0]
            end_idx = group[-1]
            count = len(group)
            
            # --- 打印信息 ---
            start_time = start_idx / fs
            duration = count / fs
            # 打印格式：通道 | 开始时间 | 持续时长 | 样本数
            print(f"[Channel {ch+1}] 时间点: {start_time:8.3f}s | 持续时长: {duration:.4f}s ({count} 个采样点)")
            
            # --- 修复数据 (线性插值) ---
            # 找到前一个正常点 (prev) 和后一个正常点 (next)
            prev_idx = start_idx - 1
            next_idx = end_idx + 1
            
            # 边界检查
            if prev_idx < 0:
                val_prev = 0 # 如果开头就是异常，设为0或取后一个值
            else:
                val_prev = cleaned_data[ch, prev_idx]
                
            if next_idx >= n_points:
                val_next = 0 # 如果结尾是异常
            else:
                val_next = cleaned_data[ch, next_idx]
            
            # 生成插值 (linspace 生成从 val_prev 到 val_next 的平滑过渡)
            # 也就是把原来这一段异常的大值，变成一条直线连接前后正常点
            # 长度需要包含这一段本身，加上首尾用于计算的点，这里只替换中间部分
            interpolated_values = np.linspace(val_prev, val_next, count + 2)[1:-1]
            
            cleaned_data[ch, group] = interpolated_values

    if total_outlier_events == 0:
        print("未发现异常值。")
    else:
        print(f"检测结束: 共修复 {total_outlier_events} 个异常片段。")
    print("==========================================\n")
    
    return cleaned_data
def save_clean_eeg(filtered_data, original_marker, new_filename):
    """
    将处理后的数据转置，拼接Marker，并保存为新文件。
    
    参数:
    filtered_data: np.array, 形状 (7, N)
    original_marker: np.array, 形状 (N,) 或 (N, 1)
    original_filename: str, 原始文件名
    """
    print(f"\n正在保存数据...")
    
    # 1. 转置数据: (Channels, Points) -> (Points, Channels)
    # 例如 (7, 12500) -> (12500, 7)
    data_transposed = filtered_data.T
    
    # 2. 确保 Marker 维度正确 (Points, 1)
    if original_marker.ndim == 1:
        marker_reshaped = original_marker[:, np.newaxis] # 变成列向量
    else:
        marker_reshaped = original_marker
        
    # 检查长度是否一致
    if data_transposed.shape[0] != marker_reshaped.shape[0]:
        print(f"错误: 数据长度 ({data_transposed.shape[0]}) 与 Marker 长度 ({marker_reshaped.shape[0]}) 不匹配！无法保存。")
        return

    # 3. 拼接数据 (Points, 7) + (Points, 1) -> (Points, 8)
    final_data = np.hstack((data_transposed, marker_reshaped))
    
    
    # 5. 保存
    np.save(new_filename, final_data)
    print(f"保存成功!")
    print(f"  文件名: {new_filename}")
    print(f"  数据形状: {final_data.shape} (Rows=TimePoints, Cols=Channels+Marker)")



def eeg_quality_check(eeg_data, fs=250):
    """
    EEG信号质量检测函数 (集成异常值报告与修复)
    """
    
    # ---------------------------------------------------------
    # 步骤 0: 去除大幅度异常值 (带日志打印)
    # ---------------------------------------------------------
    # 注意：这里传入了 fs 参数
    eeg_data_clean = remove_large_outliers(eeg_data, fs, threshold=100)


    #f0 = 50.0  
    #Q = 30.0   
    #b_notch, a_notch = signal.iirnotch(f0, Q, fs)

    #eeg_data_filtered = signal.filtfilt(b_notch, a_notch, eeg_data_bandpassed, axis=1)
    
    data = eeg_data_clean

    n_channels, n_points = data.shape
    seg_len = 250  
    n_segs = n_points // seg_len 
    
    thresh = [10, 20, 2] 
    
    res_50hz = np.zeros((n_channels, n_segs))
    res_emg = np.zeros((n_channels, n_segs))
    res_out = np.zeros((n_channels, n_segs))
    
    for ch in range(n_channels):
        for i in range(n_segs):
            start_idx = i * seg_len
            end_idx = start_idx + seg_len
            segment = data[ch, start_idx:end_idx]
            
            f, Pxx = signal.welch(segment, fs=fs, nperseg=2048)
            
            power_50hz = np.sum(Pxx[np.where((f >= 49) & (f <= 51))])
            power_emg = np.sum(Pxx[np.where((f >= 20) & (f <= 40))])
            
            # 这里的 outlier 指的是滤波后残留的小幅度伪迹 (>100)
            outlier = np.where((segment <= -100) | (segment >= 100))[0].shape[0]
            
            res_50hz[ch, i] = power_50hz
            res_emg[ch, i] = power_emg
            res_out[ch, i] = outlier

    # 绘图部分保持不变...
    fig_time, axes = plt.subplots(7, 1, figsize=(15, 12), sharex=True)
    fig_time.suptitle('EEG Time Domain Quality Check', fontsize=16)
    time_axis = np.arange(n_points) / fs
    
    for ch in range(n_channels):
        ax = axes[ch]
        ax.plot(time_axis, data[ch, :], color='black', linewidth=0.8, alpha=0.8)
        ax.set_ylabel(f'Ch {ch+1}')
        for i in range(n_segs):
            t_start = i * seg_len / fs
            t_end = (i + 1) * seg_len / fs
            p50 = res_50hz[ch, i]
            pemg = res_emg[ch, i]
            pout = res_out[ch, i]
            if p50 > thresh[0]:
                ax.axvspan(t_start, t_end, color='red', alpha=0.3, lw=0)
            elif pemg > thresh[1]:
                ax.axvspan(t_start, t_end, color='blue', alpha=0.3, lw=0)
            elif (pout > thresh[2]):
                ax.axvspan(t_start, t_end, color='gray', alpha=0.5, lw=0)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    fig_heat, axes_heat = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [res_50hz, res_emg, res_out]
    titles = [f'50Hz Power (Th={thresh[0]})', f'EMG Power (Th={thresh[1]})', f'Outlier Count (Th={thresh[2]})']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreenWhiteRed", ["#77dd77", "#ffffff", "#ff6961"])
    
    for idx, ax in enumerate(axes_heat.flat):
        current_data = metrics[idx]
        current_thresh = thresh[idx]
        d_min, d_max = np.min(current_data), np.max(current_data)
        vmin = min(d_min, current_thresh * 0.5) 
        vmax = max(d_max, current_thresh * 1.5)
        if vmin == vmax: vmin -= 0.1; vmax += 0.1
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=current_thresh, vmax=vmax)
        sns.heatmap(current_data, ax=ax, cmap=custom_cmap, norm=norm, cbar=True, cbar_kws={'label': 'Magnitude'}, yticklabels=[f'Ch{i+1}' for i in range(n_channels)])
        ax.set_title(titles[idx])
        ax.set_xlabel('Segment Index')
        ax.set_ylabel('Channel')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    return [res_50hz, res_emg, res_out], data

#第一步，去除异常值

for i in range(4):
    for type in ["TRAIN","TEST"]:
        for j in range(19):
            try:
                filename = f'D:\\EEG_Image_decode-main\\EEG_Image_decode-main\\neuradock_data\\sub-001-1000px\\sess{i+1}_run{j+1:02d}_{type}.npy'
                print(filename)
                raw_data = np.load(filename)
                marker = raw_data[:, 7] 
                # 假设数据格式处理
                if raw_data.shape[0] > 7: 
                    raw_data = raw_data[:,:-1] # 去除可能的marker列
                    raw_data = np.transpose(raw_data, (1,0))
                print("成功加载本地数据。")
                # 运行检测函数
                result_arrays, eeg_data_filtered = eeg_quality_check(raw_data, fs=250)
                # 保存清洗后的数据
                save_clean_eeg(eeg_data_filtered, marker, f'D:\\EEG_Image_decode-main\\EEG_Image_decode-main\\neuradock_data\\sub-001\\sess{i+1}_run{j+1:02d}_{type}.npy')
            except Exception as e:
                print(f"未找到文件或加载错误 ({e})，生成随机测试数据...")
                
                
                






import os
import numpy as np
import pandas as pd
import mne
import scipy
from sklearn.discriminant_analysis import _cov
from tqdm import tqdm

# ================= 1. 超参数设置 =================
ROOT_DIR = r'D:\EEG_Image_decode-main\EEG_Image_decode-main' 
SUB_ID = 'sub-001'

# 形状相关设置
PAD_TO_63_CHANNELS = False  # <--- 如果必须输出 63 通道(补0)，请改为 True；否则输出真实的 7 通道
OUTPUT_CHANNELS = 63 if PAD_TO_63_CHANNELS else 7

# 预处理参数
ORIGIN_SFREQ = 250   
TARGET_SFREQ = 250    
TMIN = -0.3           # 起始时间
TMAX = 1.1            # 结束时间 (修改为 0.8s)
                      # 总时长 1.0s (-0.2到0.8) -> 对应 250 个点

TIME_WINDOW = 300

# Train 数据结构参数
TRAIN_IMG_PER_CLASS = 10
TRAIN_REPS = 4

# Test 数据结构参数
TEST_N_CLASSES = 200
TEST_IMG_PER_CLASS = 1
TEST_REPS = 80 
# =================================================

class Args:
    def __init__(self, n_ses):
        self.n_ses = n_ses
        self.mvnn_dim = "epochs" 

def get_class_name(img_path):
    img_path = img_path.replace('\\', '/')
    parts = img_path.split('/')
    return parts[-2] if len(parts) > 1 else "unknown"

def get_image_name(img_path):
    img_path = img_path.replace('\\', '/')
    return img_path.split('/')[-1]

# ================= MVNN 函数 =================
def mvnn(args, epoched_test, epoched_train):
    whitened_test = []
    whitened_train = []
    if not epoched_test or not epoched_train: return [], []

    for s in range(args.n_ses):
        print(f"   [MVNN] Processing Session {s+1}...")
        session_data = [epoched_test[s], epoched_train[s]]
        sigma_part = np.empty((len(session_data), session_data[0].shape[2], session_data[0].shape[2]))
        
        for p in range(sigma_part.shape[0]):
            curr_data = session_data[p] 
            sigma_cond = np.empty((curr_data.shape[0], curr_data.shape[2], curr_data.shape[2]))
            
            loop_iter = range(curr_data.shape[0])
            if curr_data.shape[0] > 1000: loop_iter = tqdm(loop_iter, desc=f"Cov Part {p}", leave=False)

            for i in loop_iter:
                cond_data = curr_data[i]
                if args.mvnn_dim == "time":
                    sigma_cond[i] = np.mean([_cov(cond_data[:,:,t], shrinkage='auto') for t in range(cond_data.shape[2])], axis=0)
                elif args.mvnn_dim == "epochs":
                    if cond_data.shape[0] == 1:
                        sigma_cond[i] = _cov(np.transpose(cond_data[0]), shrinkage='auto')
                    else:
                        sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]), shrinkage='auto') for e in range(cond_data.shape[0])], axis=0)
            sigma_part[p] = sigma_cond.mean(axis=0)
        
        sigma_tot = sigma_part[1]
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

        d_test = session_data[0]
        d_flat = d_test.reshape(-1, d_test.shape[2], d_test.shape[3]) 
        d_white = (d_flat.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
        whitened_test.append(d_white.reshape(d_test.shape))
        
        d_train = session_data[1]
        d_flat_tr = d_train.reshape(-1, d_train.shape[2], d_train.shape[3])
        d_white_tr = (d_flat_tr.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
        whitened_train.append(d_white_tr.reshape(d_train.shape))

    return whitened_test, whitened_train

def process_single_session(data_dir, session_files):
    session_dict = {} 
    
    for f_npy in session_files:
        f_csv = f_npy.replace('.npy', '_events.csv')
        csv_path = os.path.join(data_dir, f_csv)
        npy_path = os.path.join(data_dir, f_npy)
        
        if not os.path.exists(csv_path): continue
        
        # 1. 加载
        data = np.load(npy_path)
        df = pd.read_csv(csv_path)
        
        eeg = data[:, :7].T*1e-6
        stim = data[:, 7].T
        stim[stim == 10000] = 0
        
        info = mne.create_info(['Ch1','Ch2','Ch3','Ch4','Ch5','Ch6','Ch7'], ORIGIN_SFREQ, ['eeg']*7)
        raw = mne.io.RawArray(eeg, info, verbose=False)
        stim_info = mne.create_info(['STI'], ORIGIN_SFREQ, ['stim'])
        raw.add_channels([mne.io.RawArray(stim.reshape(1,-1), stim_info, verbose=False)], force_update_info=True)
        
        # 2. 预处理
        raw.notch_filter(50.0, n_jobs=1, verbose=False)
        raw.filter(1, 20.0, n_jobs=1, verbose=False)
        
        if ORIGIN_SFREQ != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, verbose=False)
        
        # 3. 提取事件 & 过滤
        events = mne.find_events(raw, stim_channel='STI', output='onset', verbose=False)
        
        events = events[events[:, 2] != 9999]
        
        valid_indices = df[df['marker'] != 9999].index.tolist()


        print(len(valid_indices))
        print(valid_indices)
        
        # 4. Epoching (-0.2 到 0.8)
        # picks='eeg' 确保只取 7 个通道
        # baseline=(None, 0) 确保 -0.2~0s 被减去
        epochs = mne.Epochs(raw, events, tmin=TMIN, tmax=TMAX, 
                            baseline=(None, -0.1), 
                            preload=True)
        
        # 5. 裁剪到 250 点
        epochs.crop(tmin=-0.1, tmax=TMAX, include_tmax=False)
        
        data_epoched = epochs.get_data() # (N, 7, 250)
        data_epoched = data_epoched[:,:7,:]
        
        for k, original_idx in enumerate(valid_indices):
            img_path = df.iloc[original_idx]['image_path']
            class_name = get_class_name(img_path)
            img_name = get_image_name(img_path)
            
            key = (class_name, img_name)
            if key not in session_dict:
                session_dict[key] = []
            session_dict[key].append(data_epoched[k])
            
    return session_dict

def dict_to_array_for_mvnn(data_dict):
    sorted_keys = sorted(data_dict.keys())
    print(sorted_keys)
    max_reps = 0
    for k in sorted_keys:
        max_reps = max(max_reps, len(data_dict[k]))
    
    if max_reps == 0: return np.empty((0,0,7,TIME_WINDOW)), [] # 250点
    
    result_array = np.zeros((len(sorted_keys), max_reps, 7, TIME_WINDOW))
    
    for i, k in enumerate(sorted_keys):
        trials = data_dict[k]
        n = len(trials)
        if n > 0:
            result_array[i, :n, :, :] = np.array(trials)
    
    return result_array, sorted_keys

def main_processing():
    data_dir = os.path.join(ROOT_DIR, 'neuradock_data', SUB_ID)
    files = sorted(os.listdir(data_dir))
    sessions = sorted(list(set([f.split('_')[0] for f in files if f.startswith('sess')])))
    
    sess_test_arrays = []
    sess_train_arrays = []
    sess_test_keys = []
    sess_train_keys = []
    
    for sess in sessions:
        print(f"\n>> Reading {sess}...")
        test_files = [f for f in files if sess in f and 'TEST' in f and f.endswith('.npy')]
        train_files = [f for f in files if sess in f and 'TRAIN' in f and f.endswith('.npy')]
        
        test_dict = process_single_session(data_dir, test_files)
        train_dict = process_single_session(data_dir, train_files)
        
        test_arr, test_k = dict_to_array_for_mvnn(test_dict)
        train_arr, train_k = dict_to_array_for_mvnn(train_dict)
        
        sess_test_arrays.append(test_arr)
        sess_train_arrays.append(train_arr)
        sess_test_keys.append(test_k)
        sess_train_keys.append(train_k)

    print("\n>> Running MVNN...")
    args = Args(n_ses=len(sessions))
    white_test_list, white_train_list = mvnn(args, sess_test_arrays, sess_train_arrays)
    
    # 还原字典
    print("\n>> Reconstructing final datasets...")
    final_test_dict = {}
    final_train_dict = {}
    
    def restore_to_dict(array_data, keys, target_dict):
        for i, key in enumerate(keys): 
            class_name, img_name = key
            reps_data = array_data[i] # (Reps, 7, 250)
            for r in range(reps_data.shape[0]):
                epoch = reps_data[r]
                if not np.all(epoch == 0):
                    if class_name not in target_dict:
                        target_dict[class_name] = {}
                    if img_name not in target_dict[class_name]:
                        target_dict[class_name][img_name] = []
                    target_dict[class_name][img_name].append(epoch)

    for s in range(len(sessions)):
        restore_to_dict(white_test_list[s], sess_test_keys[s], final_test_dict)
        restore_to_dict(white_train_list[s], sess_train_keys[s], final_train_dict)

    # --- 格式化 Train: (N*10, 4, Ch, 250) ---
    print(f">> Formatting TRAIN data...")
    sorted_train_classes = sorted(final_train_dict.keys())
    
    # 临时的 5维 数组 (N_cls, 10, 4, 7, 250)
    temp_train = np.zeros((len(sorted_train_classes), TRAIN_IMG_PER_CLASS, TRAIN_REPS, 7, TIME_WINDOW))
    
    for i, c in enumerate(sorted_train_classes):
        imgs = sorted(final_train_dict[c].keys())
        current_imgs = imgs[:TRAIN_IMG_PER_CLASS]
        for j, img in enumerate(current_imgs):
            trials = final_train_dict[c][img]
            if len(trials) == 0: continue
            while len(trials) < TRAIN_REPS:
                trials.append(trials[0]) 
            temp_train[i, j, :, :, :] = np.array(trials[:TRAIN_REPS])
    
    # Flatten: 将前两个维度 (N, 10) 合并 -> (N*10, 4, 7, 250)
    final_train_array = temp_train.reshape(-1, TRAIN_REPS, 7, TIME_WINDOW)

    # --- 格式化 Test: (200, 80, Ch, 250) ---
    print(f">> Formatting TEST data...")
    sorted_test_classes = sorted(final_test_dict.keys())[:TEST_N_CLASSES]
    
    # 直接生成 (200, 80, 7, 250)
    final_test_array = np.zeros((TEST_N_CLASSES, TEST_REPS, 7, TIME_WINDOW))
    
    for i, c in enumerate(sorted_test_classes):
        imgs = sorted(final_test_dict[c].keys())
        if len(imgs) > 0:
            img_name = imgs[0]
            trials = final_test_dict[c][img_name]
            if len(trials) > 0:
                while len(trials) < TEST_REPS:
                    trials.append(trials[0]) 
                # 这里填入 (80, 7, 250)
                final_test_array[i, :, :, :] = np.array(trials[:TEST_REPS])

    # ================= 补零逻辑 (如果需要 63 通道) =================
    if PAD_TO_63_CHANNELS:
        print(">> Padding channels from 7 to 63 with zeros...")
        # Train: (Total_imgs, 4, 7, 250) -> (Total_imgs, 4, 63, 250)
        train_padded = np.zeros((final_train_array.shape[0], final_train_array.shape[1], 63, TIME_WINDOW))
        train_padded[:, :, :7, :] = final_train_array
        final_train_array = train_padded
        
        # Test: (200, 80, 7, 250) -> (200, 80, 63, 250)
        test_padded = np.zeros((final_test_array.shape[0], final_test_array.shape[1], 63, TIME_WINDOW))
        test_padded[:, :, :7, :] = final_test_array
        final_test_array = test_padded

    return final_train_array, final_test_array

if not os.path.exists(os.path.join(ROOT_DIR, 'neuradock_data', SUB_ID)):
    print("路径错误，请检查 ROOT_DIR")
else:
    train_data, test_data = main_processing()
    
    print(f"\nFinal Train Shape: {train_data.shape}") # 预期 (N*10, 4, 7or63, 250)
    print(f"Final Test Shape: {test_data.shape}")   # 预期 (200, 80, 7or63, 250)
    print(train_data)
    np.save(r"D:\EEG_Image_decode-main\EEG_Image_decode-main\neuradock_data\sub-001-preprocessed\preprocessed_eeg_training__4__1__20.npy", train_data)
    np.save(r"D:\EEG_Image_decode-main\EEG_Image_decode-main\neuradock_data\sub-001-preprocessed\preprocessed_eeg_test__4__1__20.npy", test_data)
    



    dd = np.load(r"D:\EEG_Image_decode-main\EEG_Image_decode-main\neuradock_data\sub-001-preprocessed\preprocessed_eeg_training__4__1__20.npy")
    dd[:,0,:,:] = (dd[:,0,:,:]+dd[:,1,:,:]+dd[:,2,:,:]+dd[:,3,:,:]+dd[:,4,:,:]+dd[:,5,:,:])/6
    dd[:,1,:,:] = dd[:,0,:,:]
    dd = dd[:,:2,:,:]
    np.save(r"D:\EEG_Image_decode-main\EEG_Image_decode-main\neuradock_data\sub-001-preprocessed\preprocessed_eeg_training__4__1__20_average.npy",dd)
    
    print("保存完成！")
    
    