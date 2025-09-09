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

# グローバル変数で処理結果を保持
processed_data = {
    "frames": [],
    "total_depth": 0
}

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
def plot_ratios(white_ratios, total_depth):
    num_frames = len(white_ratios)
    if num_frames > 1:
        x_data = np.linspace(0, total_depth, num=num_frames)
    else:
        x_data = [0]
    y_data = np.array(white_ratios)

    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(x_data, y_data, label='White Ratio', color='blue')

    ax.grid(True)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of White Pixels vs. Depth (Hover for values)')
    ax.legend()

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = line.get_xydata()[ind["ind"][0]]
        annot.xy = pos
        text = f"Depth: {pos[0]:.2f}m\nRatio: {pos[1]:.4f}"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.5)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show(block=False)

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
    try:
        depth = int(depth_entry.get())
        if depth <= 0:
            result_label.config(text="深度は0より大きい値を入力してください。")
            return
    except ValueError:
        result_label.config(text="深度には数値を入力してください。")
        return

    file_name = filename_entry.get()+".csv"
    depth_label = depth_entry.get()
    print(file_name)
    # 動画の総フレーム数をカウント
    total_frames = count_total_frames(video_path)
    
    # 何フレーム置きに抽出するかを決める（interval）
    interval = int(total_frames / depth * scale)
    if interval == 0:
        result_label.config(text="間隔が0です。深度またはスケールの設定を確認してください。")
        return
    
    print(f"Total frames: {total_frames}")
    print(f"Interval: {interval}")

    # フレームを抽出
    frames = extract_frames(video_path, interval)
    
    # 後の画像表示のためにフレームと深度を保存
    processed_data["frames"] = frames
    processed_data["total_depth"] = depth

    # フレームを処理して白いピクセルの割合を計算
    white_ratios = process_frames(frames, threshold, trim)

    # 結果をプロット
    plot_ratios(white_ratios, depth)

    # 結果をCSVファイルに保存
    save_ratios_to_csv(white_ratios, save_directory, file_name, interval, depth_label)
    result_label.config(text=f"処理が完了しました。CSVが {save_directory} に保存されました。")
    
    # 画像表示用のGUI要素を有効化
    depth_view_label.config(state='normal')
    depth_view_entry.config(state='normal')
    show_image_button.config(state='normal')

def show_image_at_depth():
    frames = processed_data["frames"]
    total_depth = processed_data["total_depth"]

    if not frames or total_depth == 0:
        result_label.config(text="先に動画を処理してください。")
        return

    try:
        view_depth = float(depth_view_entry.get())
    except ValueError:
        result_label.config(text="表示したい深度には数値を入力してください。")
        return

    if not (0 <= view_depth <= total_depth):
        result_label.config(text=f"深度は0から{total_depth}の間で入力してください。")
        return

    num_frames = len(frames)
    
    # 指定された深度に最も近いフレームのインデックスを計算
    if num_frames > 1:
        depth_step = total_depth / (num_frames - 1)
        idx = int(round(view_depth / depth_step))
    else:
        idx = 0
    
    # インデックスが範囲内にあることを確認
    idx = max(0, min(idx, num_frames - 1))

    original_frame = frames[idx]
    binary_frame = binarize_frame(original_frame, threshold)

    # 元の画像と二値化画像を並べて表示
    plt.figure(figsize=(12, 6))

    # 元の画像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Frame (Approx. Depth: {view_depth:.2f}m)')
    plt.axis('off')

    # 二値化画像
    plt.subplot(1, 2, 2)
    plt.imshow(binary_frame, cmap='gray')
    plt.title(f'Binarized Frame (Threshold: {threshold})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Tkinterウィンドウを設定
root = Tk()
root.title("海中浮遊物概算ツールv4")
root.geometry("400x600")

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

# 処理後に特定の深度の画像を表示するためのGUI要素
depth_view_label = Label(root, text="表示したい深度を入力 (処理後に有効化)", state='disabled')
depth_view_label.pack(pady=5)
depth_view_entry = Entry(root, state='disabled')
depth_view_entry.pack(pady=5)
show_image_button = Button(root, text="画像を表示", command=show_image_at_depth, state='disabled')
show_image_button.pack(pady=5)


# Tkinterウィンドウを開始
root.mainloop()
