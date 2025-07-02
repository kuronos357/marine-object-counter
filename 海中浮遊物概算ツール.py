import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from tkinter import filedialog, Tk

# 動画の総フレーム数をカウントする関数
def count_total_frames(fail_pass):
    cap = cv2.VideoCapture(fail_pass)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video from {fail_pass}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# ローカルファイルのパスを選択
Tk().withdraw()  # Tkinterのルートウィンドウを非表示に
fail_pass = filedialog.askopenfilename(title="動画ファイルを選択してください")

# 深度の入力を要求
depth = int(input("深度を入力: "))

# 細かさ、閾値、トリムを設定
scale = 5
threshold = 50
trim = 0

# 総フレーム数をカウント
total_frames = count_total_frames(fail_pass)
print(f"Total frames: {total_frames}")

# 何フレーム置きに抽出するかを決める
interval = int(total_frames / depth * scale)
print(f"Interval: {interval}")

def extract_frames(video_path, interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video from {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def binarize_frame(frame, threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    return binary_frame

def calculate_white_ratio(binary_frame, trim):
    total_pixels = binary_frame.size
    white_pixels = int(np.sum(binary_frame == 255)) + trim
    white_ratio = white_pixels / total_pixels
    return white_ratio

def process_frames(frames, threshold, trim=0):
    white_ratios = []
    print("フレーム数;白いピクセルの割合;閾値;フレーム番号")

    for i, frame in enumerate(frames):
        binary_frame = binarize_frame(frame, threshold)
        white_ratio = calculate_white_ratio(binary_frame, trim)
        white_ratios.append(white_ratio)

    return white_ratios

def plot_ratios(white_ratios):
    x = list(range(len(white_ratios)))
    plt.figure(figsize=(10, 5))
    plt.plot(x, white_ratios, label='White Ratio', color='blue')
    plt.grid()
    plt.xlabel('Number of frames')
    plt.ylabel('Percentage')
    plt.title('Percentage of white pixels')
    plt.legend()
    plt.show()

def save_ratios_to_csv(white_ratios, save_path, filename='color_ratios.csv'):
    with open(os.path.join(save_path, filename), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Number", "Threshold", "Interval", 'Frame', 'データだよ'])
        for i, white_ratio in enumerate(white_ratios):
            writer.writerow([i, threshold, interval, i * interval, white_ratio])

# フレームを一定間隔で抽出
frames = extract_frames(fail_pass, interval)

# フレームを処理して白いピクセルの割合を取得
white_ratios = process_frames(frames, threshold, trim)

# 白いピクセルの割合をプロット
plot_ratios(white_ratios)

# 保存先のディレクトリを選択
save_path = filedialog.askdirectory(title="保存先ディレクトリを選択")

# 白いピクセルの割合をCSVファイルに保存
save_ratios_to_csv(white_ratios, save_path)

print(f"CSVファイルが {save_path} に保存されました。")
