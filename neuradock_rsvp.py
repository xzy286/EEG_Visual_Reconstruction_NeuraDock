# import os
# import time
# import threading
# import queue
# import random
# import json
# import socket
# import numpy as np
# import pandas as pd
# import traceback
# from psychopy import visual, core, event, gui

# # ==========================================
# # 1. 实验配置 (CONFIGURATION)
# # ==========================================
# # --- 路径设置 ---
# PATH_TEST_DIR = r"D:\EEG_Image_decode-main\EEG_Image_decode-main\test_images\test_images"   
# PATH_TRAIN_DIR = r"D:\EEG_Image_decode-main\EEG_Image_decode-main\training_images\training_images" 
# PATH_TARGET_IMG = r"D:\EEG_Image_decode-main\EEG_Image_decode-main\buzz.jpg" 

# # --- 保存路径 ---
# DATA_ROOT = "./neuradock_data"

# # --- 硬件连接 ---
# EEG_IP = "198.18.0.1"
# EEG_PORT = 9600
# SAMPLING_RATE = 250 

# # --- 标记定义 ---
# MARKER_REST_START = 8000
# MARKER_REST_END   = 8001
# MARKER_TARGET     = 9999    
# MARKER_SEQ_START  = 10000 

# # --- 实验结构参数 ---
# IMGS_PER_SEQ = 20
# REST_DURATION = 300 # 5分钟

# # [Test Runs 配置]
# NUM_TEST_RUNS = 4
# SEQS_PER_TEST_RUN = 51   
# TEST_TARGETS_PER_RUN = 20 

# # [Train Runs 配置]
# NUM_TRAIN_RUNS = 15
# SEQS_PER_TRAIN_RUN = 56
# # 每一半训练集的大小 (ThingsEEG 标准是 16540 总数的一半)
# TRAIN_SET_SPLIT_SIZE = 8270 

# # --- 时间参数 (60Hz) ---
# FRAMES_IMG_ON     = 6  
# FRAMES_ISI        = 6  
# FRAMES_PRE_BLANK  = 45 
# FRAMES_POST_BLANK = 45 

# # ==========================================
# # 2. 辅助工具 (ID映射与数据准备)
# # ==========================================
# def generate_category_mapping(root_dir, start_id=1):
#     try:
#         if not os.path.exists(root_dir): raise FileNotFoundError(f"Path not found: {root_dir}")
#         subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
#         subfolders.sort()
#         mapping = {}
#         all_images = []
#         current_id = start_id
#         print(f">> Scanning {root_dir}...")
#         for folder_name in subfolders:
#             class_id = current_id
#             mapping[folder_name] = class_id
#             current_id += 1
#             folder_path = os.path.join(root_dir, folder_name)
#             for img in os.listdir(folder_path):
#                 if img.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     all_images.append({
#                         'path': os.path.join(folder_path, img),
#                         'class_id': class_id,
#                         'class_name': folder_name,
#                         'is_target': False
#                     })
#         return mapping, all_images, current_id
#     except Exception as e: print(f"[Error Mapping] {e}"); return {}, [], start_id

# def prepare_session_data(subject_dir, session_num):
#     """
#     根据 Session ID 自动决定加载哪一半训练集
#     Session 1, 3 -> Part A (前一半)
#     Session 2, 4 -> Part B (后一半)
#     """
#     print(f"\n[Data Prep] Preparing data for SESSION {session_num}...")
#     try:
#         # 1. 映射所有图片
#         map_test, imgs_test, next_id = generate_category_mapping(PATH_TEST_DIR, start_id=1)
#         map_train, imgs_train, _ = generate_category_mapping(PATH_TRAIN_DIR, start_id=next_id)
        
#         # 保存映射表
#         with open(os.path.join(subject_dir, f"session{session_num}_class_map.json"), 'w') as f:
#             json.dump({**map_test, **map_train}, f, indent=4)

#         all_runs = []

#         # ==========================
#         # Part A. Test Runs (Run 1-4)
#         # ==========================
#         # 测试集在所有 Session 中都是一样的 (200张 * 20次)
#         print(f">> Generating Test Runs (Same for all sessions)...")
#         big_test_pool = imgs_test * 20 
#         random.shuffle(big_test_pool)
        
#         imgs_per_test_run = len(big_test_pool) // NUM_TEST_RUNS
#         start_idx = 0
#         for i in range(NUM_TEST_RUNS):
#             run_id = i + 1
#             subset = big_test_pool[start_idx : start_idx + imgs_per_test_run]
#             start_idx += imgs_per_test_run
            
#             run_data = build_run_logic(
#                 image_pool=subset, 
#                 n_targets=TEST_TARGETS_PER_RUN,
#                 n_seqs=SEQS_PER_TEST_RUN,
#                 run_id=run_id,
#                 run_type="TEST",
#                 max_target_per_seq=99
#             )
#             all_runs.append(run_data)

#         # ==========================
#         # Part B. Train Runs (Run 5-19)
#         # ==========================
#         print(f">> Generating Train Runs (Splitting dataset)...")
        
#         # 1. 关键：使用固定种子打乱，确保每次运行的顺序一致
#         random.seed(42) 
#         random.shuffle(imgs_train)
        
#         # 2. 根据 Session ID 切分
#         # 如果训练集总数不够标准的一半，就用全部
#         if len(imgs_train) < TRAIN_SET_SPLIT_SIZE:
#             print("[Warning] Total images less than split size. Using ALL images.")
#             selected_train_imgs = imgs_train
#         else:
#             # 标准切分
#             split_idx = TRAIN_SET_SPLIT_SIZE # 8270
            
#             if session_num == 1 or session_num == 3:
#                 print(f"   [Session {session_num}] Using PART A (First {split_idx} images)")
#                 selected_train_imgs = imgs_train[:split_idx]
#             elif session_num == 2 or session_num == 4:
#                 print(f"   [Session {session_num}] Using PART B (Remaining images)")
#                 selected_train_imgs = imgs_train[split_idx : split_idx*2] # 取后一半
#                 # 如果图片总数不够两倍，取到末尾
#                 if len(selected_train_imgs) < split_idx:
#                     selected_train_imgs = imgs_train[split_idx:]
#             else:
#                 print(f"   [Session {session_num}] Unknown session, defaulting to Part A.")
#                 selected_train_imgs = imgs_train[:split_idx]

#         print(f"   -> Count of unique training images: {len(selected_train_imgs)}")

#         # 3. 这里的逻辑和之前一样：选中的图 * 2次，填补剩余空缺为 Target
#         # 重置随机种子以保证 Run 内部的随机性 (可选，但推荐)
#         random.seed(int(time.time())) 
        
#         standard_pool = selected_train_imgs * 2
#         random.shuffle(standard_pool)
        
#         # 计算需要的 Target 数
#         total_slots = NUM_TRAIN_RUNS * SEQS_PER_TRAIN_RUN * IMGS_PER_SEQ # 16800
#         num_standard = len(standard_pool)
#         num_targets_needed = total_slots - num_standard
        
#         print(f"   - Standard Slots: {num_standard}")
#         print(f"   - Target Slots needed: {num_targets_needed}")
        
#         if num_targets_needed > (NUM_TRAIN_RUNS * SEQS_PER_TRAIN_RUN):
#              raise ValueError("Too many targets required, cannot satisfy Max 1 per seq.")

#         # 4. 分配 Target (保证每序列最多1个)
#         total_train_seqs = NUM_TRAIN_RUNS * SEQS_PER_TRAIN_RUN
#         seq_has_target = [True] * num_targets_needed + [False] * (total_train_seqs - num_targets_needed)
#         random.shuffle(seq_has_target)
        
#         pool_idx = 0
#         all_train_sequences = []
        
#         for has_target in seq_has_target:
#             current_seq = []
#             if has_target:
#                 current_seq.append({'path': PATH_TARGET_IMG, 'class_id': MARKER_TARGET, 'class_name': 'target', 'is_target': True})
#                 slots_needed = IMGS_PER_SEQ - 1
#             else:
#                 slots_needed = IMGS_PER_SEQ
            
#             for _ in range(slots_needed):
#                 if pool_idx < len(standard_pool):
#                     current_seq.append(standard_pool[pool_idx])
#                     pool_idx += 1
#                 else:
#                     current_seq.append(selected_train_imgs[0].copy()) # Fallback
            
#             random.shuffle(current_seq)
#             all_train_sequences.append(current_seq)
            
#         # 5. 分配给 Run
#         start_seq_idx = 0
#         for i in range(NUM_TRAIN_RUNS):
#             run_id = NUM_TEST_RUNS + i + 1
#             run_seqs = all_train_sequences[start_seq_idx : start_seq_idx + SEQS_PER_TRAIN_RUN]
#             start_seq_idx += SEQS_PER_TRAIN_RUN
#             all_runs.append({'run_id': run_id, 'type': 'TRAIN', 'sequences': run_seqs})

#         return all_runs
#     except Exception as e: print(f"[Error Prep] {e}"); traceback.print_exc(); return []

# def build_run_logic(image_pool, n_targets, n_seqs, run_id, run_type, max_target_per_seq=99):
#     """通用 Run 构建逻辑 (用于 Test Set)"""
#     targets_dist = [0] * n_seqs
#     for _ in range(n_targets):
#         while True:
#             idx = random.randint(0, n_seqs-1)
#             if targets_dist[idx] < max_target_per_seq:
#                 targets_dist[idx] += 1
#                 break
#     sequences = []
#     pool_idx = 0
#     for seq_i in range(n_seqs):
#         num_targets_here = targets_dist[seq_i]
#         num_normal_imgs = IMGS_PER_SEQ - num_targets_here
#         current_seq = []
#         for _ in range(num_targets_here):
#             current_seq.append({'path': PATH_TARGET_IMG, 'class_id': MARKER_TARGET, 'class_name': 'target', 'is_target': True})
#         for _ in range(num_normal_imgs):
#             if pool_idx < len(image_pool):
#                 current_seq.append(image_pool[pool_idx])
#                 pool_idx += 1
#             else:
#                 current_seq.append({'path':PATH_TARGET_IMG, 'class_id':0, 'is_target':False})
#         random.shuffle(current_seq)
#         sequences.append(current_seq)
#     return {'run_id': run_id, 'type': run_type, 'sequences': sequences}

# # ==========================================
# # 3. 硬件与多线程 (保持不变)
# # ==========================================
# class DataStream:
#     def __init__(self, IP, PORT, buffer_size=1024, total_channels=8, used_channels=7, pkg_groups=1, data_group_len=1):
#         self.ip = IP; self.port = PORT; self.buffer_size = buffer_size
#         self.total_channels = total_channels; self.used_channels = used_channels
#         self.pkg_groups = pkg_groups; self.data_group_len = data_group_len
#         self.is_running = False; self.socket = None; self._buffer_str = ""; self._data_buffer = []

#     def __iter__(self):
#         if self.is_running: self.close()
#         self.is_running = True; self._connect(); self._buffer_str = ""; self._data_buffer = []; return self
    
#     def _connect(self):
#         try:
#             self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.socket.settimeout(5)
#             self.socket.connect((self.ip, self.port))
#             self.socket.send(b'start')
#             print(f"[DataStream] Connected.")
#         except Exception as e: print(f"[Connect Error] {e}"); self.is_running=False; raise

#     def close(self):
#         self.is_running = False; 
#         if self.socket: 
#             try: self.socket.close() 
#             except: pass; 
#             self.socket=None

#     def __next__(self):
#         if not self.is_running: raise StopIteration
#         while len(self._data_buffer) < self.data_group_len:
#             try:
#                 chunk = self.socket.recv(self.buffer_size)
#                 if not chunk: raise ConnectionError("Closed")
#                 self._buffer_str += chunk.decode('utf-8', errors='ignore')
#                 while True:
#                     lines = self._buffer_str.split('\n')
#                     if len(lines) < 2: break
#                     complete, self._buffer_str = lines[:-1], lines[-1]
#                     for line in complete:
#                         if not line.strip(): continue
#                         fields = line.strip().split(',')
#                         if len(fields) < 2 + self.pkg_groups * self.total_channels: continue
#                         try:
#                             data_vals = list(map(float, fields[2:2 + self.pkg_groups * self.total_channels]))
#                             arr = np.array(data_vals, dtype=np.float32).reshape(self.pkg_groups, self.total_channels)
#                             for t in range(self.pkg_groups): self._data_buffer.append(arr[t, :self.used_channels].tolist())
#                         except: continue
#             except socket.timeout: continue
#             except Exception: self.close(); raise StopIteration
#         res = self._data_buffer[:self.data_group_len]
#         self._data_buffer = self._data_buffer[self.data_group_len:]
#         return res

# class EEGThreadManager:
#     def __init__(self):
#         self.data_queue = queue.Queue(maxsize=100000)
#         self.stop_event = threading.Event(); self.stream = None; self.thread = None; self.current_trigger = 0

#     def start_stream(self):
#         try:
#             self.stream = DataStream(EEG_IP, EEG_PORT)
#             self.stop_event.clear()
#             self.thread = threading.Thread(target=self._worker, daemon=True)
#             self.thread.start(); return True
#         except Exception as e: print(f"[Thread Error] {e}"); return False

#     def _worker(self):
#         print(">> Worker Running.")
#         try:
#             for data_group in self.stream:
#                 if self.stop_event.is_set(): break
#                 if data_group is None: continue
#                 arr = np.array(data_group)
#                 marker = np.full((arr.shape[0], 1), self.current_trigger)
#                 self.data_queue.put(np.hstack([arr, marker]))
#         except: pass
#         finally: print(">> Worker Stopped.")
    
#     def stop_stream(self):
#         self.stop_event.set()
#         if self.stream: self.stream.close()
#         if self.thread: self.thread.join(timeout=1.0)

#     def flush_to_buffer(self, buf):
#         while not self.data_queue.empty(): 
#             try: buf.append(self.data_queue.get_nowait())
#             except queue.Empty: break

#     def set_trigger(self, code): self.current_trigger = int(code)

# # ==========================================
# # 4. 实验流程执行 (保持不变)
# # ==========================================
# def run_resting_state(win, stims, eeg_mgr, eeg_buffer, duration, label):
#     (fix_out, fix_in, _, text_stim) = stims
#     print(f">> Resting State: {label}")
#     text_stim.text = f"Resting State ({label})\n\nRelax & Fixate Center"; text_stim.draw(); win.flip()
#     event.waitKeys(keyList=['space'])
#     text_stim.text = "+"; text_stim.draw(); win.flip()
    
#     eeg_mgr.set_trigger(MARKER_REST_START); win.callOnFlip(eeg_mgr.set_trigger, 0)
#     timer = core.CountdownTimer(duration)
#     while timer.getTime() > 0:
#         fix_out.draw(); fix_in.draw(); win.flip()
#         eeg_mgr.flush_to_buffer(eeg_buffer)
#         if event.getKeys(keyList=['escape']): raise KeyboardInterrupt("User Escape")
#     eeg_mgr.set_trigger(MARKER_REST_END); win.callOnFlip(eeg_mgr.set_trigger, 0); win.flip()

# def run_rsvp_run(win, stims, run_data, eeg_mgr, eeg_buffer):
#     (fix_out, fix_in, img_stim, text_stim) = stims
#     sequences = run_data['sequences']
#     for seq_imgs in sequences:
#         # Pre-Blank
#         eeg_mgr.set_trigger(0)
#         for _ in range(FRAMES_PRE_BLANK):
#             fix_out.draw(); fix_in.draw(); win.flip(); eeg_mgr.flush_to_buffer(eeg_buffer)
#         # Sequence
#         eeg_mgr.set_trigger(MARKER_SEQ_START); win.callOnFlip(eeg_mgr.set_trigger, 0)
#         for img_info in seq_imgs:
#             img_stim.image = img_info['path']
#             trig = MARKER_TARGET if img_info['is_target'] else img_info['class_id']
#             for f in range(FRAMES_IMG_ON):
#                 img_stim.draw(); fix_out.draw(); fix_in.draw()
#                 if f == 0: win.callOnFlip(eeg_mgr.set_trigger, trig)
#                 elif f == 1: win.callOnFlip(eeg_mgr.set_trigger, 0)
#                 win.flip(); eeg_mgr.flush_to_buffer(eeg_buffer)
#             eeg_mgr.set_trigger(0)
#             for _ in range(FRAMES_ISI):
#                 fix_out.draw(); fix_in.draw(); win.flip(); eeg_mgr.flush_to_buffer(eeg_buffer)
#         # Post-Blank
#         for _ in range(FRAMES_POST_BLANK):
#             fix_out.draw(); fix_in.draw(); win.flip(); eeg_mgr.flush_to_buffer(eeg_buffer)
#         # Response
#         text_stim.text = "?"; text_stim.draw(); win.flip()
#         clock = core.Clock()
#         while clock.getTime() < 1.5: 
#             if event.getKeys(keyList=['escape']): raise KeyboardInterrupt("User Escape")
#             eeg_mgr.flush_to_buffer(eeg_buffer)

# def save_buffer(buffer, filepath):
#     try:
#         if len(buffer) > 0:
#             data = np.vstack(buffer)
#             np.save(filepath, data)
#             print(f"   [Saved] {filepath}")
#             buffer.clear()
#         else: print("   [Warning] Buffer empty.")
#     except Exception as e: print(f"[Save Error] {e}")

# # ==========================================
# # 5. 主程序 (入口)
# # ==========================================
# def run_experiment():
#     eeg_mgr = None; win = None; eeg_buffer = []
#     try:
#         # 1. 登记 - 这里增加了 Session 的输入
#         info = {'Subject': '001', 'Session': '1'}
#         dlg = gui.DlgFromDict(info, title='ThingsEEG - Final'); 
#         if not dlg.OK: return
        
#         # 获取 Session ID
#         try:
#             sess_num = int(info['Session'])
#         except ValueError:
#             print("[Error] Session must be a number (1, 2, 3, or 4).")
#             return

#         subject_dir = os.path.join(DATA_ROOT, f"sub-{info['Subject']}")
#         if not os.path.exists(subject_dir): os.makedirs(subject_dir)

#         # 2. 准备数据 (传入 session_num)
#         session_runs = prepare_session_data(subject_dir, sess_num)
#         if not session_runs: return

#         # 3. 启动硬件
#         eeg_mgr = EEGThreadManager()
#         if not eeg_mgr.start_stream(): return

#         # 4. 界面
#         win = visual.Window([1024, 768], fullscr=True, units='pix', color=[0,0,0]); win.mouseVisible = False
#         fix_out = visual.Circle(win, radius=10, lineColor='black') 
#         fix_in = visual.Circle(win, radius=5, fillColor='red', lineColor=None)
#         img_stim = visual.ImageStim(win, size=[500, 500])
#         text_stim = visual.TextStim(win, height=40)
#         stims = (fix_out, fix_in, img_stim, text_stim)

#         text_stim.text = f"Session {sess_num} Ready\n\nPress Space"; text_stim.draw(); win.flip()
#         event.waitKeys(keyList=['space'])

#         # Start Resting
#         run_resting_state(win, stims, eeg_mgr, eeg_buffer, REST_DURATION, "START")
#         save_buffer(eeg_buffer, os.path.join(subject_dir, f"sess{sess_num}_resting_start.npy"))

#         # Run Loop
#         for run_data in session_runs:
#             rid = run_data['run_id']; rtype = run_data['type']
#             text_stim.text = f"Run {rid} / 19 ({rtype})\n\nPress Space"; text_stim.draw(); win.flip()
#             event.waitKeys(keyList=['space'])
            
#             run_rsvp_run(win, stims, run_data, eeg_mgr, eeg_buffer)
#             save_buffer(eeg_buffer, os.path.join(subject_dir, f"sess{sess_num}_run{rid:02d}_{rtype}.npy"))
            
#             if rid < 19:
#                 text_stim.text = "Relax."; text_stim.draw(); win.flip(); core.wait(5.0)

#         # End Resting
#         run_resting_state(win, stims, eeg_mgr, eeg_buffer, REST_DURATION, "END")
#         save_buffer(eeg_buffer, os.path.join(subject_dir, f"sess{sess_num}_resting_end.npy"))
        
#         text_stim.text = "Finished!"; text_stim.draw(); win.flip(); core.wait(3.0)

#     except KeyboardInterrupt: print("\n[Abort] Stopped by user.")
#     except Exception as e: print(f"\n[Error] {e}"); traceback.print_exc()
#     finally:
#         if len(eeg_buffer) > 0:
#             try: np.save(os.path.join(DATA_ROOT, f"crash_{int(time.time())}.npy"), np.vstack(eeg_buffer))
#             except: pass
#         if eeg_mgr: eeg_mgr.stop_stream()
#         if win: win.close()
#         core.quit()

# if __name__ == "__main__":
#     run_experiment()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuradock RSVP Experiment
Adapted from tgro5258's script
Combined with DataStream acquisition
"""

from psychopy import core, event, visual, gui
import os, random, sys, math, json, socket, threading, queue
import pandas as pd
import numpy as np
import traceback
import time

# ==========================================
# 1. Configuration
# ==========================================
debug_testsubject = 0  # Set to 1 for quick testing
debug_windowedmode = 0 # Set to 1 for windowed mode
debug_save_screenshots = 0

# Paths
PATH_TEST_DIR = r"D:\EEG_Image_decode-main\EEG_Image_decode-main\test_images\test_images"   
PATH_TRAIN_DIR = r"D:\EEG_Image_decode-main\EEG_Image_decode-main\training_images\training_images" 
PATH_TARGET_IMG = r"D:\EEG_Image_decode-main\EEG_Image_decode-main\buzz.jpg" 
DATA_ROOT = "./neuradock_data"


# Hardware
EEG_IP = "192.168.56.1"
EEG_PORT = 9600
SAMPLING_RATE = 250 

# Timing (60Hz monitor assumed)
refreshrate = 60
fixationduration = 0.75 - .5/refreshrate
stimduration = 0.1 - .5/refreshrate
isiduration = 0.2 - .5/refreshrate # SOA 200ms -> 100ms ON, 100ms OFF. Here we use 0.2 as total cycle from stimon

FRAME_DURATION = 1.0 / refreshrate # 单帧时长，约4.17ms
STIM_FRAMES = int(round(0.1 / FRAME_DURATION)) # 100ms刺激 ≈ 24帧 (24 * 4.17ms = 100.08ms)
ISI_FRAMES = int(round(0.1 / FRAME_DURATION))  # 100ms空屏 ≈ 24帧
PRE_BLANK_FRAMES = int(round(0.75 / FRAME_DURATION)) # 750ms
POST_BLANK_FRAMES = int(round(0.75 / FRAME_DURATION))

# Markers
MARKER_REST_START = 8000
MARKER_REST_END   = 8001
MARKER_TARGET     = 9999    
MARKER_SEQ_START  = 10000 

# Experiment Structure
NUM_TEST_RUNS = 4
SEQS_PER_TEST_RUN = 51   
TEST_TARGETS_PER_RUN = 20 

NUM_TRAIN_RUNS = 15
SEQS_PER_TRAIN_RUN = 56
TRAIN_SET_SPLIT_SIZE = 8270 

IMGS_PER_SEQ = 20
REST_DURATION = 30
# ==========================================
# 2. Hardware Interface (DataStream)
# ==========================================
class DataStream:
    def __init__(self, IP, PORT, buffer_size=1024, total_channels=8, used_channels=7, pkg_groups=1, data_group_len=1):
        self.ip = IP; self.port = PORT; self.buffer_size = buffer_size
        self.total_channels = total_channels; self.used_channels = used_channels
        self.pkg_groups = pkg_groups; self.data_group_len = data_group_len
        self.is_running = False; self.socket = None; self._buffer_str = ""; self._data_buffer = []

    def __iter__(self):
        if self.is_running: self.close()
        self.is_running = True; self._connect(); self._buffer_str = ""; self._data_buffer = []; return self
    
    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.ip, self.port))
            self.socket.send(b'start')
            print(f"[DataStream] Connected to {self.ip}:{self.port}")
        except Exception as e: print(f"[DataStream] Connect Error: {e}"); self.is_running=False; raise

    def close(self):
        self.is_running = False; 
        if self.socket: 
            try: self.socket.close() 
            except: pass; 
            self.socket=None

    def __next__(self):
        if not self.is_running: raise StopIteration
        while len(self._data_buffer) < self.data_group_len:
            try:
                chunk = self.socket.recv(self.buffer_size)
                if not chunk: raise ConnectionError("Closed")
                self._buffer_str += chunk.decode('utf-8', errors='ignore')
                while True:
                    lines = self._buffer_str.split('\n')
                    if len(lines) < 2: break
                    complete, self._buffer_str = lines[:-1], lines[-1]
                    for line in complete:
                        if not line.strip(): continue
                        fields = line.strip().split(',')
                        if len(fields) < 2 + self.pkg_groups * self.total_channels: continue
                        try:
                            data_vals = list(map(float, fields[2:2 + self.pkg_groups * self.total_channels]))
                            arr = np.array(data_vals, dtype=np.float32).reshape(self.pkg_groups, self.total_channels)
                            for t in range(self.pkg_groups): self._data_buffer.append(arr[t, :self.used_channels].tolist())
                        except: continue
            except socket.timeout: continue
            except Exception: self.close(); raise StopIteration
        res = self._data_buffer[:self.data_group_len]
        self._data_buffer = self._data_buffer[self.data_group_len:]
        return res

class EEGThreadManager:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=100000)
        self.stop_event = threading.Event(); self.stream = None; self.thread = None; self.current_trigger = 0

    def start_stream(self):
        try:
            self.stream = DataStream(EEG_IP, EEG_PORT)
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start(); return True
        except Exception as e: print(f"[Thread Error] {e}"); return False

    def _worker(self):
        print(">> Worker Running.")
        try:
            for data_group in self.stream:
                if self.stop_event.is_set(): break
                if data_group is None: continue
                arr = np.array(data_group)
                marker = np.full((arr.shape[0], 1), self.current_trigger)
                self.data_queue.put(np.hstack([arr, marker]))
        except: pass
        finally: print(">> Worker Stopped.")
    
    def stop_stream(self):
        self.stop_event.set()
        if self.stream: self.stream.close()
        if self.thread: self.thread.join(timeout=1.0)

    def flush_to_buffer(self, buf):
        while not self.data_queue.empty(): 
            try: buf.append(self.data_queue.get_nowait())
            except queue.Empty: break

    def set_trigger(self, code): self.current_trigger = int(code)

# ==========================================
# 3. Helper Functions
# ==========================================
def generate_category_mapping(root_dir, start_id=1):
    if not os.path.exists(root_dir): return {}, [], start_id
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    subfolders.sort()
    mapping = {}
    all_images = []
    current_id = start_id
    print(f">> Scanning {root_dir}...")
    for folder_name in subfolders:
        class_id = current_id
        mapping[folder_name] = class_id
        current_id += 1
        folder_path = os.path.join(root_dir, folder_name)
        for img in os.listdir(folder_path):
            if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_images.append({
                    'path': os.path.join(folder_path, img),
                    'class_id': class_id,
                    'class_name': folder_name,
                    'is_target': False
                })
    return mapping, all_images, current_id

def build_run_logic(image_pool, n_targets, n_seqs, run_id, run_type):
    """Builds sequences for a run"""
    targets_dist = [0] * n_seqs
    # Distribute targets randomly (max 1 per seq logic is handled by caller or chance)
    # For Test Runs, we just distribute 20 targets across 51 sequences
    target_indices = random.sample(range(n_seqs), n_targets)
    for idx in target_indices: targets_dist[idx] += 1
    
    sequences = []
    pool_idx = 0
    
    for seq_i in range(n_seqs):
        num_targets_here = targets_dist[seq_i]
        num_normal_imgs = IMGS_PER_SEQ - num_targets_here
        current_seq = []
        for _ in range(num_targets_here):
            current_seq.append({'path': PATH_TARGET_IMG, 'class_id': MARKER_TARGET, 'class_name': 'target', 'is_target': True})
        for _ in range(num_normal_imgs):
            if pool_idx < len(image_pool):
                current_seq.append(image_pool[pool_idx])
                pool_idx += 1
            else:
                # Padding if pool exhausted
                current_seq.append({'path': PATH_TARGET_IMG, 'class_id': 0, 'is_target': False}) 
        random.seed(int(time.time()));random.shuffle(current_seq)
        sequences.append(current_seq)
    
    # Return as DataFrame for consistency with reference code style
    # Actually, reference code uses a flat eventlist. We will stick to list of lists for structure, then flatten if needed.
    return {'run_id': run_id, 'type': run_type, 'sequences': sequences}

def prepare_data(subject_dir, session_num):
    print(f"\n[Data Prep] Preparing Session {session_num}...")
    map_test, imgs_test, next_id = generate_category_mapping(PATH_TEST_DIR, start_id=1)
    map_train, imgs_train, _ = generate_category_mapping(PATH_TRAIN_DIR, start_id=next_id)
    
    with open(os.path.join(subject_dir, f"session{session_num}_class_map.json"), 'w') as f:
        json.dump({**map_test, **map_train}, f, indent=4)

    all_runs = []
    
    # --- Test Runs ---
    big_test_pool = imgs_test * 20 
    random.seed(int(time.time())); random.shuffle(big_test_pool) # Consistent shuffle per subject
    imgs_per_test_run = len(big_test_pool) // NUM_TEST_RUNS
    
    start_idx = 0
    for i in range(NUM_TEST_RUNS):
        subset = big_test_pool[start_idx : start_idx + imgs_per_test_run]
        start_idx += imgs_per_test_run
        all_runs.append(build_run_logic(subset, TEST_TARGETS_PER_RUN, SEQS_PER_TEST_RUN, i+1, "TEST"))

    # --- Train Runs ---
    random.seed(42); random.shuffle(imgs_train) # Fixed seed for splitting
    
    split_idx = TRAIN_SET_SPLIT_SIZE
    if session_num == 1 or session_num == 3:
        selected_train_imgs = imgs_train[:split_idx]
    else:
        selected_train_imgs = imgs_train[split_idx : split_idx*2]
        if len(selected_train_imgs) < split_idx: selected_train_imgs = imgs_train[split_idx:]
    
    # Reset seed for run generation
    random.seed(int(time.time()))
    standard_pool = selected_train_imgs * 2
    random.shuffle(standard_pool)
    
    total_train_seqs = NUM_TRAIN_RUNS * SEQS_PER_TRAIN_RUN
    total_slots = total_train_seqs * IMGS_PER_SEQ
    num_targets_needed = total_slots - len(standard_pool)
    
    seq_has_target = [True] * num_targets_needed + [False] * (total_train_seqs - num_targets_needed)
    random.shuffle(seq_has_target)
    
    pool_idx = 0
    all_train_sequences = []
    
    for has_target in seq_has_target:
        current_seq = []
        if has_target:
            current_seq.append({'path': PATH_TARGET_IMG, 'class_id': MARKER_TARGET, 'class_name': 'target', 'is_target': True})
            slots_needed = IMGS_PER_SEQ - 1
        else:
            slots_needed = IMGS_PER_SEQ
        
        for _ in range(slots_needed):
            if pool_idx < len(standard_pool):
                current_seq.append(standard_pool[pool_idx])
                pool_idx += 1
            else:
                current_seq.append(selected_train_imgs[0].copy())
        
        random.shuffle(current_seq)
        all_train_sequences.append(current_seq)
        
    start_seq_idx = 0
    for i in range(NUM_TRAIN_RUNS):
        run_seqs = all_train_sequences[start_seq_idx : start_seq_idx + SEQS_PER_TRAIN_RUN]
        start_seq_idx += SEQS_PER_TRAIN_RUN
        all_runs.append({'run_id': NUM_TEST_RUNS + i + 1, 'type': 'TRAIN', 'sequences': run_seqs})
        
    return all_runs



# def prepare_data(subject_dir, session_num):
#     print(f"\n[Data Prep] Preparing Session {session_num} with Recovery Logic...")
#     # 1. 生成基础映射 (和原来一样)
#     map_test, imgs_test, next_id = generate_category_mapping(PATH_TEST_DIR, start_id=1)
#     map_train, imgs_train, _ = generate_category_mapping(PATH_TRAIN_DIR, start_id=next_id)
    
#     # 保存 map (如果是为了恢复，可以不覆盖，但覆盖也没事，因为内容是一样的)
#     with open(os.path.join(subject_dir, f"session{session_num}_class_map.json"), 'w') as f:
#         json.dump({**map_test, **map_train}, f, indent=4)

#     all_runs = []
    
#     # ==========================
#     # A. Test Runs (1-4)
#     # ==========================
#     # 因为你已经跑完了 Run 1-4，这里的逻辑不重要，只要占位即可，为了保持 run_id 的连续性
#     # 我们随便生成一下，反正会被主循环跳过
#     big_test_pool = imgs_test * 20 
#     random.shuffle(big_test_pool)
#     imgs_per_test_run = len(big_test_pool) // NUM_TEST_RUNS
#     start_idx = 0
#     for i in range(NUM_TEST_RUNS):
#         # 即使是随机的也无所谓，因为我们会跳过这些 Run
#         subset = big_test_pool[start_idx : start_idx + imgs_per_test_run]
#         start_idx += imgs_per_test_run
#         all_runs.append(build_run_logic(subset, TEST_TARGETS_PER_RUN, SEQS_PER_TEST_RUN, i+1, "TEST"))

#     # ==========================
#     # B. Train Runs (5-19)
#     # ==========================
#     # 1. 重建原始的完整训练池 (和原来一样的逻辑)
#     random.seed(42) # 必须保持这个种子和原来一样
#     random.shuffle(imgs_train)
    
#     split_idx = TRAIN_SET_SPLIT_SIZE
#     if session_num == 1 or session_num == 3:
#         selected_train_imgs = imgs_train[:split_idx]
#     else:
#         selected_train_imgs = imgs_train[split_idx : split_idx*2]
#         if len(selected_train_imgs) < split_idx: selected_train_imgs = imgs_train[split_idx:]
    
#     # 这是本该在这个 Session 显示的所有图片的列表 (每张2次)
#     full_pool_list = selected_train_imgs * 2
    
#     # -------------------------------------------------
#     # 2. 核心修复逻辑：从 CSV 中读取已看过的图片
#     # -------------------------------------------------
#     print(">> Checking for completed runs to exclude used images...")
    
#     # 假设你完成了 Run 5 到 Run 9 (共5个训练Run)
#     # 我们需要从 full_pool_list 中移除这些图片
    
#     finished_train_runs = range(5, 10) # Run 5, 6, 7, 8, 9
    
#     for rid in finished_train_runs:
#         csv_path = os.path.join(subject_dir, f"sess{session_num}_run{rid:02d}_TRAIN_events.csv")
#         if os.path.exists(csv_path):
#             print(f"   - Loading {csv_path} to remove used images")
#             try:
#                 df = pd.read_csv(csv_path)
#                 # 过滤掉 Target (Target图片不从池子里扣，因为Target是额外的)
#                 # 根据你的代码，Target的 class_id 是 9999
#                 normal_trials = df[df['class_id'] != MARKER_TARGET]
                
#                 used_paths = normal_trials['image_path'].tolist()
                
#                 # 从 full_pool_list 中移除这些路径
#                 for p in used_paths:
#                     # 我们需要找到 full_pool_list 中对应的图片对象并移除一个
#                     # 因为 full_pool_list 是 dict 列表，我们比对 path
#                     found = False
#                     for i, img_obj in enumerate(full_pool_list):
#                         if img_obj['path'] == p:
#                             full_pool_list.pop(i) # 移除找到的第一个
#                             found = True
#                             break
#                     if not found:
#                         print(f"Warning: Image {p} from CSV not found in original pool! (Possible duplicate removal or mismatch)")
                        
#             except Exception as e:
#                 print(f"Error reading CSV {csv_path}: {e}")
#                 input("Critical Error in recovery logic. Press Ctrl+C to stop or Enter to ignore (RISKY).")
#         else:
#             print(f"Warning: Expected CSV {csv_path} not found. Assuming this run was NOT actually finished?")
#             # 如果 CSV 不存在，说明这轮可能没跑完，就不扣除图片，留给后面跑

#     # 3. 剩下的图片就是 Run 10-19 需要跑的
#     remaining_pool = full_pool_list
#     random.seed(int(time.time())) # 现在可以随机打乱剩下的了
#     random.shuffle(remaining_pool)
    
#     print(f">> Remaining images for Runs 10-19: {len(remaining_pool)}")
    
#     # 4. 构建剩下的 Runs (Run 10 - 19)
#     # 还需要跑多少轮训练？
#     remaining_runs_count = NUM_TRAIN_RUNS - len(finished_train_runs) # 15 - 5 = 10
    
#     # 将 remaining_pool 分成 10 份
#     # 注意：可能不能整除，需要处理余数（通常原逻辑是用 padding，我们这里尽量均匀分配）
#     chunk_size = len(remaining_pool) // remaining_runs_count
    
#     # 我们需要先为 Run 5-9 占位 (填充空数据)，保持索引一致
#     # 这样 all_runs[8] 对应 Run 9，all_runs[9] 对应 Run 10
#     for i in range(len(finished_train_runs)):
#         # 填充一个空的 run 字典，因为我们会跳过它，所以内容不重要
#         all_runs.append({'run_id': 5 + i, 'type': 'TRAIN', 'sequences': []})

#     # 生成真正的 Run 10-19
#     pool_idx = 0
#     # 为剩下的每一轮生成序列
#     for i in range(remaining_runs_count):
#         current_run_id = 10 + i
        
#         # 取出这一轮的图片
#         if i == remaining_runs_count - 1:
#             run_imgs = remaining_pool[pool_idx:] # 最后一轮拿走所有剩下的
#         else:
#             run_imgs = remaining_pool[pool_idx : pool_idx + chunk_size]
#             pool_idx += chunk_size
            
#         # 下面这段逻辑是模仿你原来的 build_run_logic，但是适配 Train
#         # 原代码 Train 逻辑比较复杂(seq_has_target)，我们简化重构一下以确保健壮性：
        
#         # 每一轮的 Sequence 数量
#         n_seqs = SEQS_PER_TRAIN_RUN
        
#         # 这一轮总共需要的 slots
#         total_slots_needed = n_seqs * IMGS_PER_SEQ
        
#         # 计算需要多少个 target 来填充空位
#         n_targets = total_slots_needed - len(run_imgs)
#         if n_targets < 0: n_targets = 0 # 理论上不应该发生
        
#         # 构造序列分配
#         # 1. 先把所有图片（run_imgs）和 Target 放在一个大列表里
#         run_stimuli = run_imgs.copy()
#         for _ in range(n_targets):
#              run_stimuli.append({'path': PATH_TARGET_IMG, 'class_id': MARKER_TARGET, 'class_name': 'target', 'is_target': True})
        
#         random.shuffle(run_stimuli)
        
#         # 2. 切分成 sequences
#         sequences = []
#         stim_idx = 0
#         for s in range(n_seqs):
#             seq = []
#             for k in range(IMGS_PER_SEQ):
#                 if stim_idx < len(run_stimuli):
#                     seq.append(run_stimuli[stim_idx])
#                     stim_idx += 1
#                 else:
#                     # 万一不够了（极少见），补第一张
#                     seq.append(run_imgs[0])
#             sequences.append(seq)
            
#         all_runs.append({'run_id': current_run_id, 'type': 'TRAIN', 'sequences': sequences})
        
#     return all_runs








# ==========================================
# 4. Main Experiment Logic
# ==========================================
if debug_testsubject:
    subjectnr = 0
    sessionnr = 1
else:
    subject_info = {'Subject number':'0001', 'Session number': '1'}
    if not gui.DlgFromDict(subject_info, title='Neuradock RSVP').OK:
        print('User hit cancel')
        exit()
    try:
        subjectnr = int(subject_info['Subject number'])
        sessionnr = int(subject_info['Session number'])
    except:
        raise

# Setup Directory
subject_dir = os.path.join(DATA_ROOT, f"sub-{subjectnr:03d}")
if not os.path.exists(subject_dir): os.makedirs(subject_dir)

# Prepare Data Structure
session_runs = prepare_data(subject_dir, sessionnr)

# Start Hardware
eeg_mgr = EEGThreadManager()
if not eeg_mgr.start_stream():
    print("EEG Stream failed to start")
    exit()
eeg_buffer = []

def save_buffer(buffer, filepath):
    if len(buffer) > 0:
        try:
            data = np.vstack(buffer)
            np.save(filepath, data)
            print(f"   [Saved] {filepath}")
            buffer.clear()
        except Exception as e:
            print(f"   [Save Error] {e}")
# === 【补充】 定义中断检查函数 ===
def check_abort(k):
    """检查用户是否按下了退出键"""
    if k:
        # 兼容 timeStamped=True (返回元组) 和 False (返回字符串)
        key = k[0][0] if isinstance(k[0], (tuple, list)) else k[0]
        if key in ['q', 'escape']:
            print(">> User requested abort.")
            raise Exception('User pressed q/escape')
try:
    if debug_windowedmode:
        win = visual.Window([800,600], screen=1, units='pix')
    else:
        win = visual.Window([1500,1500], screen=1,  fullscr=True,units='pix', color=[0,0,0])
    rsvp_stim_pool = [visual.ImageStim(win, size=1000, name=f'stim_{i}') for i in range(20)]
    mouse = event.Mouse(visible=False)

    # Creating Stimuli (Using ShapeStim for Fixation to avoid loading pngs if not present)
    # Fixation: Black Outer Circle, Red Inner Dot
    fix_out = visual.Circle(win, radius=10, lineColor='black', fillColor='black') 
    fix_in = visual.Circle(win, radius=5, fillColor='red', lineColor=None)
    
    # If target is present, we might want a different fixation (reference code logic)
    # But ThingsEEG paper says "bull's eye fixation throughout".
    # We will stick to one fixation style.
    
    querytext = visual.TextStim(win, text='', pos=(0,200))
    progresstext = visual.TextStim(win, text='', pos=(0,100))
    sequencestarttext = visual.TextStim(win, text='', pos=(0,50))
    
    # Image Stimulus Placeholder
    # Loading textures on the fly is risky for timing. 
    # Reference code loaded "stimtex = []" before the sequence loop. We will do the same.
    
    def loadstimtex(stimname):
        return visual.ImageStim(win, stimname, size=500, name=os.path.basename(stimname))
    
    # -------------------------------------------------------------------------
    # RESTING STATE (START)
    # -------------------------------------------------------------------------
    sequencestarttext.text = f"Session {sessionnr}\nResting State (START)\nRelax & Fixate"
    sequencestarttext.draw(); win.flip()
    event.waitKeys(keyList=['space'])
    
    sequencestarttext.text = "+"; sequencestarttext.draw()
    eeg_mgr.set_trigger(MARKER_REST_START)
    time_rest_start = win.flip()
    # Pulse trigger
    win.callOnFlip(eeg_mgr.set_trigger, 0)
    
    while core.getTime() < time_rest_start + REST_DURATION:
        fix_out.draw(); fix_in.draw(); win.flip()
        eeg_mgr.flush_to_buffer(eeg_buffer)
        if event.getKeys(keyList=['escape']): raise Exception('User Escape')
    
    eeg_mgr.set_trigger(MARKER_REST_END)
    win.flip()
    save_buffer(eeg_buffer, os.path.join(subject_dir, f"sess{sessionnr}_resting_start.npy"))

    # -------------------------------------------------------------------------
    # RUN LOOP
    # -------------------------------------------------------------------------
    for run_data in session_runs:
        rid = run_data['run_id']
        rtype = run_data['type']
        run_events = [] # 初始化事件列表
        progresstext.text = f"Run {rid} / 19 ({rtype})"
        sequencestarttext.text = "Press SPACE to start run"
        progresstext.draw(); sequencestarttext.draw(); win.flip()
        
        event.waitKeys(keyList=['space'])
        
        sequences = run_data['sequences']
        nsequences = len(sequences)
        for seq_idx, seq_imgs in enumerate(sequences):
            for i, img_info in enumerate(seq_imgs):
                # stim = stimtex[i]
                stim = rsvp_stim_pool[i]
                # Logic for trigger
                trig = MARKER_TARGET if img_info['is_target'] else img_info['class_id']
                run_events.append({
                        'seq_idx': seq_idx,
                        'img_idx': i,
                        'image_path': img_info['path'],
                        'class_id': img_info['class_id'],
                        'marker': trig
                    })
        for seq_idx, seq_imgs in enumerate(sequences):
            
            # 1. Pre-load textures for this sequence (Critical for timing)
            # stimtex = []
            # for img_info in seq_imgs:
            #     stimtex.append(loadstimtex(img_info['path']))
            for i, img_info in enumerate(seq_imgs):
                # 把路径赋值给 image 属性，PsychoPy 会自动加载
                rsvp_stim_pool[i].image = img_info['path']
                # 确保它是不透明的 (以防万一)
                rsvp_stim_pool[i].opacity = 1.0
            # 2. Sequence Info Screen (Optional, reference code had it per seq, ThingsEEG usually does continuous)
            # We will follow ThingsEEG continuous RSVP within a run. 
            # Reference code pauses per sequence. We will pause ONLY if needed or follow ThingsEEG timing strictly.
            # ThingsEEG: "Every rapid serial sequence started with 750ms blank... then 20 images... then 750ms blank... then 2s response"
            # It implies continuous flow.
            
            # A. Pre-Blank (750ms)
            eeg_mgr.set_trigger(0)
            fix_out.draw(); fix_in.draw()
            time_fixon = win.flip()
            while core.getTime() < time_fixon + fixationduration:
                eeg_mgr.flush_to_buffer(eeg_buffer)
            
            # B. RSVP Loop
            eeg_mgr.set_trigger(MARKER_SEQ_START)
            # We use win.callOnFlip to set trigger exactly at first frame
            
            for i, img_info in enumerate(seq_imgs):
                # stim = stimtex[i]
                stim = rsvp_stim_pool[i]
                # Logic for trigger
                trig = MARKER_TARGET if img_info['is_target'] else img_info['class_id']
                
                # # Draw Stim
                # stim.draw(); fix_out.draw(); fix_in.draw()
                
                # # Set trigger on flip
                # eeg_mgr.set_trigger(trig)
                # time_stimon = win.flip() # FLIP 1: Stim ON
                
                # # Reset trigger immediately for pulse effect (next frame or manual delay?)
                # # Since we loop fast, we can set trigger to 0 after flip in the loop
                # # But to ensure it lasts at least 1 frame, we set it to 0 before next flip
                
                

                
                
                
                
                
                # while core.getTime() < time_stimon + stimduration:
                #     eeg_mgr.flush_to_buffer(eeg_buffer)
                #     # Reset trigger logic: 
                #     # If we want pulse, we can set to 0 here. 
                #     # But thread picks it up asynchronously. 
                #     # Ideally, wait 10ms then set 0.
                #     if core.getTime() - time_stimon > 0.01: 
                #         eeg_mgr.set_trigger(0)
                
                # # Draw Fixation (ISI)
                # fix_out.draw(); fix_in.draw()
                # time_stimoff = win.flip() # FLIP 2: Stim OFF
                
                # # Wait ISI
                # # Calculate next onset based on SOA (isiduration here is SOA or Gap?)
                # # Reference: "SOA of 200ms" -> 100 on, 100 off.
                # # stimduration is 0.1.
                # # Next ON should be time_stimon + 0.2
                
                # while core.getTime() < time_stimon + 0.2: 
                #     eeg_mgr.flush_to_buffer(eeg_buffer)
            
            
            
                # ---- 刺激呈现阶段 (精确24帧) ----
                for frame in range(STIM_FRAMES):
                    stim.draw(); fix_out.draw(); fix_in.draw()
                    if frame == 0: # 只在第一帧设置刺激标记
                        win.callOnFlip(eeg_mgr.set_trigger, trig)
                    elif frame == 3: # 在第二帧立刻清零，实现一个超短脉冲（约4ms）
                        win.callOnFlip(eeg_mgr.set_trigger, 0)
                    win.flip()
                    eeg_mgr.flush_to_buffer(eeg_buffer)
                
                # ---- 空屏(ISI)阶段 (精确24帧) ----
                for _ in range(ISI_FRAMES):
                    fix_out.draw(); fix_in.draw()
                    win.flip()
                    eeg_mgr.flush_to_buffer(eeg_buffer)
            # C. Post-Blank (750ms)
            # fix_out.draw(); fix_in.draw()
            # time_postblank = win.flip()
            # while core.getTime() < time_postblank + fixationduration:
            #     eeg_mgr.flush_to_buffer(eeg_buffer)
            
            # C. Post-Blank
            for _ in range(POST_BLANK_FRAMES):
                fix_out.draw(); fix_in.draw()
                win.flip()
                eeg_mgr.flush_to_buffer(eeg_buffer)    
            
            # D. Response Window (Max 2s)
            querytext.text = "?"
            querytext.draw()
            time_query = win.flip()
            
            response = None
            while core.getTime() < time_query + 0.2*random.random()+1.8:
                keys = event.getKeys(keyList=['y', 'n', 'space'], timeStamped=True)
                eeg_mgr.flush_to_buffer(eeg_buffer)
                if keys:
                    check_abort(keys)
                    response = keys[0][0]
                    break
        
        # End of Run
        save_buffer(eeg_buffer, os.path.join(subject_dir, f"sess{sessionnr}_run{rid:02d}_{rtype}.npy"))
        
        pd.DataFrame(run_events).to_csv(os.path.join(subject_dir, f"sess{sessionnr}_run{rid:02d}_{rtype}_events.csv"), index=False)
        
        # Break
        if rid < 19:
            sequencestarttext.text = "Run Finished. Relax."
            sequencestarttext.draw(); win.flip()
            core.wait(5.0)

    # -------------------------------------------------------------------------
    # RESTING STATE (END)
    # -------------------------------------------------------------------------
    sequencestarttext.text = "Resting State (END)\nRelax & Fixate"
    sequencestarttext.draw(); win.flip()
    event.waitKeys(keyList=['space'])
    
    sequencestarttext.text = "+"; sequencestarttext.draw()
    eeg_mgr.set_trigger(MARKER_REST_START)
    time_rest_start = win.flip()
    win.callOnFlip(eeg_mgr.set_trigger, 0)
    
    while core.getTime() < time_rest_start + REST_DURATION:
        fix_out.draw(); fix_in.draw(); win.flip()
        eeg_mgr.flush_to_buffer(eeg_buffer)
    
    eeg_mgr.set_trigger(MARKER_REST_END)
    win.flip()
    save_buffer(eeg_buffer, os.path.join(subject_dir, f"sess{sessionnr}_resting_end.npy"))

finally:
    print(str(sys.exc_info()))
    if 'sequencestarttext' in locals():
        sequencestarttext.text = 'Experiment finished!'
        sequencestarttext.draw()
        win.flip()
    core.wait(1)
    
    # Save any leftovers
    if 'eeg_buffer' in locals() and len(eeg_buffer) > 0:
         try: 
             np.save(os.path.join(subject_dir, f"crash_{int(time.time())}.npy"), np.vstack(eeg_buffer))
         except: pass
         
    if 'eeg_mgr' in locals(): eeg_mgr.stop_stream()
    if 'win' in locals(): win.close()
    exit()