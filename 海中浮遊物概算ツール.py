import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from tkinter import filedialog, Tk, Label, Button, Entry

# 設定値
scale = 5  # フレーム抽出の細かさ
threshold = 50  # 二値化の閾値
trim = 0  # トリムの値

# 動画の総フレーム数をカウントする関数
def count_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video from {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# フレームを一定間隔で抽出する関数
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

# フレームをグレースケールに変換して二値化する関数
def binarize_frame(frame, threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    return binary_frame

# 二値化されたフレームの白いピクセルの割合を計算する関数
def calculate_white_ratio(binary_frame, trim):
    total_pixels = binary_frame.size
    white_pixels = int(np.sum(binary_frame == 255)) + trim
    white_ratio = white_pixels / total_pixels
    return white_ratio

# 全てのフレームを処理して白いピクセルの割合を計算する関数
def process_frames(frames, threshold, trim=0):
    white_ratios = []
    for frame in frames:
        binary_frame = binarize_frame(frame, threshold)
        white_ratio = calculate_white_ratio(binary_frame, trim)
        white_ratios.append(white_ratio)
    return white_ratios

# 白いピクセルの割合をプロットする関数
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

# 白いピクセルの割合をCSVファイルに保存する関数
def save_ratios_to_csv(white_ratios, save_path, filename, interval, depth_label):
    with open(os.path.join(save_path, filename), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Number", "Threshold", "Interval", 'Frame', depth_label])
        for i, white_ratio in enumerate(white_ratios):
            writer.writerow([i, threshold, interval, i * interval, white_ratio])


# GUI用関数
def select_video_file():
    video_path = filedialog.askopenfilename(title="動画ファイルを選択してください")
    video_label.config(text=video_path)
    return video_path

def select_save_directory():
    save_directory = filedialog.askdirectory(title="保存先ディレクトリを選択")
    save_label.config(text=save_directory)
    return save_directory

def process_video():
    video_path = video_label.cget("text")
    save_directory = save_label.cget("text")
    depth = int(depth_entry.get())
    file_name = filename_entry.get()+".csv"
    depth_label = depth_entry.get()
    print(file_name)
    # 動画の総フレーム数をカウント
    total_frames = count_total_frames(video_path)
    
    # 何フレーム置きに抽出するかを決める（interval）
    interval = int(total_frames / depth * scale)
    
    print(f"Total frames: {total_frames}")
    print(f"Interval: {interval}")

    # フレームを抽出
    frames = extract_frames(video_path, interval)

    # フレームを処理して白いピクセルの割合を計算
    white_ratios = process_frames(frames, threshold, trim)

    # 結果をプロット
    plot_ratios(white_ratios)

    # 結果をCSVファイルに保存
    save_ratios_to_csv(white_ratios, save_directory, file_name, interval, depth_label)
    result_label.config(text=f"処理が完了しました。CSVが {save_directory} に保存されました。")

# Tkinterウィンドウを設定
root = Tk()
root.title("海中浮遊物概算ツールv3")
root.geometry("400x500")

# GUI要素
Label(root, text="深度を入力").pack(pady=5)
depth_entry = Entry(root)
depth_entry.pack(pady=5)

Button(root, text="動画ファイルを選択", command=select_video_file).pack(pady=5)
video_label = Label(root, text="動画ファイルが選択されていません")
video_label.pack(pady=5)

Label(root, text="保存するファイル名を入力").pack(pady=5)
filename_entry = Entry(root)
filename_entry.pack(pady=5)

Button(root, text="保存先ディレクトリを選択", command=select_save_directory).pack(pady=5)
save_label = Label(root, text="保存先ディレクトリが選択されていません")
save_label.pack(pady=5)

Button(root, text="動画を処理", command=process_video).pack(pady=20)
result_label = Label(root, text="")
result_label.pack(pady=5)

# Tkinterウィンドウを開始
root.mainloop()
