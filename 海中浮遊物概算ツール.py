import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from tkinter import filedialog, Tk, Label, Button, Entry, Radiobutton, StringVar, Frame

# 設定値
scale = 5  # フレーム抽出の細かさ
threshold = 50  # 二値化の閾値
trim = 0  # トリムの値

# グローバル変数で処理結果を保持
processed_data = {
    "frames": [],
    "total_depth": 0,
    "fps": 0,
    "interval": 0,
    "total_frames": 0
}

# 動画のプロパティ(総フレーム数, FPS)を取得する関数
def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video from {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

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
    
    # 動画のプロパティを取得
    total_frames, fps = get_video_properties(video_path)
    if fps == 0:
        result_label.config(text="動画のFPSが取得できませんでした。")
        return

    # 何フレーム置きに抽出するかを決める（interval）
    interval = int(total_frames / depth * scale)
    if interval == 0:
        result_label.config(text="間隔が0です。深度またはスケールの設定を確認してください。")
        return
    
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"Interval: {interval}")

    # フレームを抽出
    frames = extract_frames(video_path, interval)
    
    # 後の画像表示のために各種データを保存
    processed_data["frames"] = frames
    processed_data["total_depth"] = depth
    processed_data["fps"] = fps
    processed_data["interval"] = interval
    processed_data["total_frames"] = total_frames

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
    for rb in radio_buttons:
        rb.config(state='normal')

def show_image():
    frames = processed_data["frames"]
    if not frames:
        result_label.config(text="先に動画を処理してください。")
        return

    mode = input_mode.get()
    value_str = depth_view_entry.get()
    idx = -1
    title_info = ""

    num_frames = len(frames)
    total_depth = processed_data["total_depth"]
    interval = processed_data["interval"]
    fps = processed_data["fps"]
    total_frames_original = processed_data["total_frames"]

    try:
        if mode == 'depth':
            view_depth = float(value_str)
            if not (0 <= view_depth <= total_depth):
                result_label.config(text=f"深度は0から{total_depth}の間で入力してください。")
                return
            if num_frames > 1:
                depth_step = total_depth / (num_frames - 1)
                idx = int(round(view_depth / depth_step))
            else:
                idx = 0
            title_info = f"Approx. Depth: {view_depth:.2f}m"

        elif mode == 'frame':
            view_frame_num = int(value_str)
            if not (0 <= view_frame_num < num_frames):
                result_label.config(text=f"フレーム番号は0から{num_frames - 1}の間で入力してください。")
                return
            idx = view_frame_num
            title_info = f"Frame Index: {view_frame_num}"

        elif mode == 'seconds':
            view_seconds = float(value_str)
            duration = total_frames_original / fps
            if not (0 <= view_seconds <= duration):
                result_label.config(text=f"秒数は0から{duration:.2f}の間で入力してください。")
                return
            original_frame_num = view_seconds * fps
            idx = int(round(original_frame_num / interval))
            title_info = f"Approx. Time: {view_seconds:.2f}s"

    except ValueError:
        result_label.config(text="有効な数値を入力してください。")
        return

    idx = max(0, min(idx, num_frames - 1))

    original_frame = frames[idx]
    binary_frame = binarize_frame(original_frame, threshold)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Frame ({title_info})')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(binary_frame, cmap='gray')
    plt.title(f'Binarized Frame (Threshold: {threshold})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Tkinterウィンドウを設定
root = Tk()
root.title("海中浮遊物概算ツールv5")
root.geometry("550x800")

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
depth_view_label = Label(root, text="表示方法を選択し、値を入力", state='disabled')
depth_view_label.pack(pady=5)

input_mode = StringVar(value="depth")
radio_frame = Frame(root)
radio_frame.pack()

radio_buttons = [
    Radiobutton(radio_frame, text="深度(m)", variable=input_mode, value="depth", state='disabled'),
    Radiobutton(radio_frame, text="ﾌﾚｰﾑ番号", variable=input_mode, value="frame", state='disabled'),
    Radiobutton(radio_frame, text="秒数(s)", variable=input_mode, value="seconds", state='disabled')
]
for rb in radio_buttons:
    rb.pack(side='left', padx=5)

depth_view_entry = Entry(root, state='disabled')
depth_view_entry.pack(pady=5)
show_image_button = Button(root, text="画像を表示", command=show_image, state='disabled')
show_image_button.pack(pady=5)


# Tkinterウィンドウを開始
root.mainloop()
