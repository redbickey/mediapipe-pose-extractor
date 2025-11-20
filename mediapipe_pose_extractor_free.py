"""
MediaPipe Pose Extractor - 無料版
画像から骨格データを抽出するツール
※ 動画処理機能は有料版でのみ利用可能

Author: Shintaro
Version: Free 1.0
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageFile
from pathlib import Path
import threading
import time

# ドラッグ＆ドロップ用
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("⚠️ tkinterdnd2がインストールされていません。ドラッグ＆ドロップは無効です。")
    print("   インストール: pip install tkinterdnd2")

# PILで大きな画像や切り詰められた画像を確実に読み込む
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def imwrite_unicode(filename, img):
    """日本語パスに対応した画像書き込み"""
    try:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        success, encoded_img = cv2.imencode('.png', img)
        if not success:
            print(f"⚠️ 画像のエンコードに失敗: {filename}")
            return False
        with open(filename, 'wb') as f:
            f.write(encoded_img.tobytes())
        return True
    except Exception as e:
        print(f"⚠️ 画像書き込みエラー ({filename}): {e}")
        return False

# ----------------------------------------------------------------------
# MediaPipe 定義と接続データ
# ----------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.NOSE),
    (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.NOSE),
]

# デフォルトの色設定
DEFAULT_POSE_COLORS = {
    mp_pose.PoseLandmark.NOSE: (0, 0, 255), mp_pose.PoseLandmark.LEFT_EYE: (255, 0, 0),
    mp_pose.PoseLandmark.RIGHT_EYE: (255, 0, 0), mp_pose.PoseLandmark.LEFT_EAR: (255, 0, 0),
    mp_pose.PoseLandmark.RIGHT_EAR: (255, 0, 0), mp_pose.PoseLandmark.LEFT_SHOULDER: (255, 170, 0),
    mp_pose.PoseLandmark.LEFT_ELBOW: (255, 85, 0), mp_pose.PoseLandmark.LEFT_WRIST: (255, 0, 0),
    mp_pose.PoseLandmark.RIGHT_SHOULDER: (0, 0, 255), mp_pose.PoseLandmark.RIGHT_ELBOW: (0, 85, 255),
    mp_pose.PoseLandmark.RIGHT_WRIST: (0, 170, 255), mp_pose.PoseLandmark.LEFT_HIP: (0, 255, 0),
    mp_pose.PoseLandmark.RIGHT_HIP: (0, 255, 0), mp_pose.PoseLandmark.LEFT_KNEE: (85, 255, 0),
    mp_pose.PoseLandmark.LEFT_ANKLE: (170, 255, 0), mp_pose.PoseLandmark.LEFT_FOOT_INDEX: (255, 255, 0),
    mp_pose.PoseLandmark.RIGHT_KNEE: (255, 0, 170), mp_pose.PoseLandmark.RIGHT_ANKLE: (255, 0, 85),
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: (255, 0, 0),
}

POSE_MAP_MP_TO_OP = {
    0: mp_pose.PoseLandmark.NOSE, 2: mp_pose.PoseLandmark.RIGHT_SHOULDER, 3: mp_pose.PoseLandmark.RIGHT_ELBOW, 
    4: mp_pose.PoseLandmark.RIGHT_WRIST, 5: mp_pose.PoseLandmark.LEFT_SHOULDER, 6: mp_pose.PoseLandmark.LEFT_ELBOW,     
    7: mp_pose.PoseLandmark.LEFT_WRIST, 9: mp_pose.PoseLandmark.RIGHT_HIP, 10: mp_pose.PoseLandmark.RIGHT_KNEE, 
    11: mp_pose.PoseLandmark.RIGHT_ANKLE, 12: mp_pose.PoseLandmark.LEFT_HIP, 13: mp_pose.PoseLandmark.LEFT_KNEE,      
    14: mp_pose.PoseLandmark.LEFT_ANKLE, 15: mp_pose.PoseLandmark.RIGHT_EYE, 16: mp_pose.PoseLandmark.LEFT_EYE, 
    17: mp_pose.PoseLandmark.RIGHT_EAR, 18: mp_pose.PoseLandmark.LEFT_EAR, 19: mp_pose.PoseLandmark.LEFT_FOOT_INDEX, 
    22: mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, 
}

HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
FACE_CONNECTIONS = list(mp_face_mesh.FACEMESH_TESSELATION)

# ----------------------------------------------------------------------
# コア処理ロジック
# ----------------------------------------------------------------------
def draw_colored_pose_from_lm(pose_image, pose_results, visibility_threshold, h, w, op_keypoints=None, 
                               line_thickness=4, point_radius=6, pose_colors=None, use_custom_color=False, custom_color=(255, 255, 255)):
    if pose_colors is None:
        pose_colors = DEFAULT_POSE_COLORS
    
    for lm, color in pose_colors.items():
        if lm.value < len(pose_results.pose_landmarks.landmark):
            landmark = pose_results.pose_landmarks.landmark[lm.value]
            if landmark.visibility >= visibility_threshold:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                final_color = custom_color if use_custom_color else color
                cv2.circle(pose_image, (x, y), point_radius, final_color, -1)
    
    for connection in POSE_CONNECTIONS:
        start_lm = connection[0]
        end_lm = connection[1]
        if start_lm.value < len(pose_results.pose_landmarks.landmark) and end_lm.value < len(pose_results.pose_landmarks.landmark):
            start = pose_results.pose_landmarks.landmark[start_lm.value]
            end = pose_results.pose_landmarks.landmark[end_lm.value]
            if start.visibility >= visibility_threshold and end.visibility >= visibility_threshold:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                if use_custom_color:
                    line_color = custom_color
                else:
                    start_color = pose_colors.get(start_lm, (255, 255, 255))
                    end_color = pose_colors.get(end_lm, (255, 255, 255))
                    line_color = tuple((np.array(start_color) + np.array(end_color)) // 2)
                    line_color = tuple(int(c) for c in line_color)
                
                cv2.line(pose_image, start_point, end_point, line_color, line_thickness)
    
    if op_keypoints:
        for i, (x, y, c) in enumerate(op_keypoints):
            if c > 0:
                color = custom_color if use_custom_color else pose_colors.get(POSE_MAP_MP_TO_OP.get(i), (255, 255, 255))
                cv2.circle(pose_image, (int(x), int(y)), point_radius, color, -1)

def process_single_image(input_path, output_dir, mode, complexity, visibility, 
                         line_thickness, point_radius, background_color, use_custom_color, 
                         custom_color, single_color_mode, log_func=None):
    try:
        # 画像読み込み
        img_pil = Image.open(input_path)
        if img_pil.mode == 'RGBA':
            img_pil = img_pil.convert('RGB')
        image = np.array(img_pil)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 骨格画像生成
        pose_image = np.full((h, w, 3), background_color, dtype=np.uint8)
        
        # MediaPipe処理
        if mode == "Full Control (統合)":
            with mp_pose.Pose(static_image_mode=True, model_complexity=complexity, 
                              min_detection_confidence=0.5) as pose:
                with mp_hands.Hands(static_image_mode=True, max_num_hands=2, 
                                     min_detection_confidence=0.5) as hands:
                    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                                min_detection_confidence=0.5) as face_mesh:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(image_rgb)
                        hand_results = hands.process(image_rgb)
                        face_results = face_mesh.process(image_rgb)
                        
                        # オーバーレイ画像の生成
                        overlay = image.copy()
                        
                        # Pose描画
                        if pose_results.pose_landmarks:
                            draw_colored_pose_from_lm(pose_image, pose_results, visibility, h, w, 
                                                       line_thickness=line_thickness, point_radius=point_radius,
                                                       use_custom_color=single_color_mode, custom_color=custom_color)
                            mp_drawing.draw_landmarks(overlay, pose_results.pose_landmarks, 
                                                       mp_pose.POSE_CONNECTIONS)
                        
                        # Hands描画
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                for landmark in hand_landmarks.landmark:
                                    x = int(landmark.x * w)
                                    y = int(landmark.y * h)
                                    color = custom_color if single_color_mode else (0, 255, 0)
                                    cv2.circle(pose_image, (x, y), max(1, point_radius//2), color, -1)
                                mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Face描画
                        if face_results.multi_face_landmarks:
                            for face_landmarks in face_results.multi_face_landmarks:
                                for i, landmark in enumerate(face_landmarks.landmark):
                                    if i % 5 == 0:  # 間引いて描画
                                        x = int(landmark.x * w)
                                        y = int(landmark.y * h)
                                        color = custom_color if single_color_mode else (255, 255, 0)
                                        cv2.circle(pose_image, (x, y), max(1, point_radius//3), color, -1)
                                mp_drawing.draw_landmarks(overlay, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                        
                        # JSON生成
                        json_data = {"pose": None, "hands": [], "face": None}
                        
                        if pose_results.pose_landmarks:
                            json_data["pose"] = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                                                  for lm in pose_results.pose_landmarks.landmark]
                        
                        if hand_results.multi_hand_landmarks:
                            json_data["hands"] = [[{"x": lm.x, "y": lm.y, "z": lm.z} 
                                                    for lm in hand_landmarks.landmark]
                                                   for hand_landmarks in hand_results.multi_hand_landmarks]
                        
                        if face_results.multi_face_landmarks:
                            json_data["face"] = [{"x": lm.x, "y": lm.y, "z": lm.z}
                                                  for lm in face_results.multi_face_landmarks[0].landmark]
                        
        elif mode == "Simple Pose (簡易)":
            with mp_pose.Pose(static_image_mode=True, model_complexity=complexity, 
                              min_detection_confidence=0.5) as pose:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(image_rgb)
                
                overlay = image.copy()
                
                if pose_results.pose_landmarks:
                    draw_colored_pose_from_lm(pose_image, pose_results, visibility, h, w, 
                                               line_thickness=line_thickness, point_radius=point_radius,
                                               use_custom_color=single_color_mode, custom_color=custom_color)
                    mp_drawing.draw_landmarks(overlay, pose_results.pose_landmarks, 
                                               mp_pose.POSE_CONNECTIONS)
                
                json_data = {"pose": None}
                if pose_results.pose_landmarks:
                    json_data["pose"] = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                                          for lm in pose_results.pose_landmarks.landmark]
        
        elif mode == "Pose + Hands":
            with mp_pose.Pose(static_image_mode=True, model_complexity=complexity, 
                              min_detection_confidence=0.5) as pose:
                with mp_hands.Hands(static_image_mode=True, max_num_hands=2, 
                                     min_detection_confidence=0.5) as hands:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(image_rgb)
                    hand_results = hands.process(image_rgb)
                    
                    overlay = image.copy()
                    
                    if pose_results.pose_landmarks:
                        draw_colored_pose_from_lm(pose_image, pose_results, visibility, h, w, 
                                                   line_thickness=line_thickness, point_radius=point_radius,
                                                   use_custom_color=single_color_mode, custom_color=custom_color)
                        mp_drawing.draw_landmarks(overlay, pose_results.pose_landmarks, 
                                                   mp_pose.POSE_CONNECTIONS)
                    
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            for landmark in hand_landmarks.landmark:
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                color = custom_color if single_color_mode else (0, 255, 0)
                                cv2.circle(pose_image, (x, y), max(1, point_radius//2), color, -1)
                            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    json_data = {"pose": None, "hands": []}
                    
                    if pose_results.pose_landmarks:
                        json_data["pose"] = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                                              for lm in pose_results.pose_landmarks.landmark]
                    
                    if hand_results.multi_hand_landmarks:
                        json_data["hands"] = [[{"x": lm.x, "y": lm.y, "z": lm.z} 
                                                for lm in hand_landmarks.landmark]
                                               for hand_landmarks in hand_results.multi_hand_landmarks]
        
        # 結果の保存
        base_name = Path(input_path).stem
        
        # 骨格画像を保存
        pose_path = os.path.join(output_dir, f"{base_name}_pose.png")
        imwrite_unicode(pose_path, pose_image)
        
        # オーバーレイ画像を保存
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        imwrite_unicode(overlay_path, overlay)
        
        # JSONを保存
        json_path = os.path.join(output_dir, f"{base_name}_pose.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if log_func:
            log_func(f"✅ 処理完了: {base_name}")
        
        return True
        
    except Exception as e:
        if log_func:
            log_func(f"❌ エラー: {str(e)}")
        return False

# ----------------------------------------------------------------------
# メインGUIアプリケーション（無料版）
# ----------------------------------------------------------------------
class PoseExtractorAppFree:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe Pose Extractor - Free Version")
        
        # 変数初期化
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value="./output_poses")
        self.mode = tk.StringVar(value="Full Control (統合)")
        self.complexity = tk.IntVar(value=2)
        self.visibility = tk.DoubleVar(value=0.0)
        self.line_thickness = tk.IntVar(value=4)
        self.point_radius = tk.IntVar(value=6)
        self.background_color = (0, 0, 0)
        self.custom_color = (255, 255, 255)
        self.single_color_mode = tk.BooleanVar(value=False)
        self.overlay_display = tk.BooleanVar(value=False)
        
        self.batch_files = []
        self.processing_thread = None
        self.is_processing = False
        
        # GUI構築
        self.setup_ui()
        
        # ドラッグ＆ドロップ設定
        if HAS_DND:
            self.setup_drag_drop()
    
    def setup_drag_drop(self):
        """ドラッグ＆ドロップの設定"""
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
    
    def on_drop(self, event):
        """ドロップ時の処理"""
        files = self.root.tk.splitlist(event.data)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        image_files = []
        for file in files:
            path = Path(file)
            if path.suffix.lower() in image_extensions:
                image_files.append(str(path))
            elif path.is_dir():
                for ext in image_extensions:
                    image_files.extend([str(p) for p in path.glob(f"*{ext}")])
                    image_files.extend([str(p) for p in path.glob(f"*{ext.upper()}")])
        
        if image_files:
            if len(image_files) == 1:
                self.input_file.set(image_files[0])
                self.batch_files = []
                self.batch_label.config(text="")
                self.show_image_preview(image_files[0])
            else:
                self.batch_files = image_files
                self.input_file.set(image_files[0])
                self.batch_label.config(text=f"バッチ処理: {len(image_files)}ファイル選択中")
                self.show_image_preview(image_files[0])
        else:
            messagebox.showwarning("警告", "有効な画像ファイルがありません")
    
    def setup_ui(self):
        """UIの構築"""
        # 上部にバージョン情報
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        version_label = ttk.Label(info_frame, text="無料版 - 画像処理機能のみ", font=("", 10, "bold"), foreground="blue")
        version_label.pack(side=tk.LEFT)
        
        upgrade_btn = ttk.Button(info_frame, text="🚀 完全版を入手", command=self.show_upgrade_info)
        upgrade_btn.pack(side=tk.RIGHT)
        
        # メインコンテンツ
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 画像処理タブのコンテンツ
        self.create_image_tab(main_frame)
    
    def show_upgrade_info(self):
        """アップグレード情報を表示"""
        msg = """MediaPipe Pose Extractor 完全版の機能：

✅ 画像処理（無料版と同じ）
✅ 動画処理機能
  - 動画から骨格データを抽出
  - フレーム毎のPNG/JSON出力
  - 骨格動画の生成
  - バッチ処理対応

完全版は有料記事でダウンロード可能です。
詳細は記事をご覧ください。"""
        
        messagebox.showinfo("完全版について", msg)
    
    def create_image_tab(self, parent):
        """画像処理タブの作成"""
        # 上部コントロール
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 入力ファイル
        file_frame = ttk.LabelFrame(control_frame, text="入力ファイル (ドラッグ＆ドロップ対応)", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        file_row = ttk.Frame(file_frame)
        file_row.pack(fill=tk.X)
        
        ttk.Entry(file_row, textvariable=self.input_file, width=50).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(file_row, text="参照...", command=self.browse_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_row, text="複数選択", command=self.browse_multiple_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_row, text="フォルダ", command=self.browse_image_folder).pack(side=tk.LEFT, padx=2)
        
        self.batch_label = ttk.Label(file_frame, text="", foreground="blue")
        self.batch_label.pack(anchor=tk.W, pady=(2, 0))
        
        # 基本設定
        settings_frame = ttk.LabelFrame(control_frame, text="基本設定", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 5))
        
        settings_row1 = ttk.Frame(settings_frame)
        settings_row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(settings_row1, text="モード:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(settings_row1, textvariable=self.mode, 
                                  values=["Full Control (統合)", "Simple Pose (簡易)", "Pose + Hands"],
                                  state="readonly", width=18)
        mode_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(settings_row1, text="精度:").pack(side=tk.LEFT)
        ttk.Scale(settings_row1, from_=0, to=2, variable=self.complexity, 
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=5)
        self.complexity_label = ttk.Label(settings_row1, text="2", width=3)
        self.complexity_label.pack(side=tk.LEFT)
        self.complexity.trace_add("write", self.update_complexity_label)
        
        ttk.Label(settings_row1, text="閾値:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Scale(settings_row1, from_=0.0, to=1.0, variable=self.visibility,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=5)
        
        self.visibility_entry = ttk.Entry(settings_row1, width=6)
        self.visibility_entry.pack(side=tk.LEFT)
        self.visibility_entry.insert(0, "0.00")
        self.visibility_entry.bind('<Return>', self.on_visibility_entry)
        self.visibility_entry.bind('<FocusOut>', self.on_visibility_entry)
        self.visibility.trace_add("write", self.update_visibility_from_slider)
        
        settings_row2 = ttk.Frame(settings_frame)
        settings_row2.pack(fill=tk.X, pady=2)
        
        ttk.Label(settings_row2, text="保存先:").pack(side=tk.LEFT)
        ttk.Entry(settings_row2, textvariable=self.output_dir, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(settings_row2, text="参照...", command=self.browse_output_dir).pack(side=tk.LEFT)
        
        # 描画設定
        draw_frame = ttk.LabelFrame(control_frame, text="描画設定", padding="5")
        draw_frame.pack(fill=tk.X, pady=(0, 5))
        
        draw_row1 = ttk.Frame(draw_frame)
        draw_row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(draw_row1, text="線の太さ:").pack(side=tk.LEFT)
        ttk.Scale(draw_row1, from_=1, to=10, variable=self.line_thickness,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=5)
        self.line_thickness_label = ttk.Label(draw_row1, text="4", width=3)
        self.line_thickness_label.pack(side=tk.LEFT)
        self.line_thickness.trace_add("write", lambda *a: self.line_thickness_label.config(text=str(int(self.line_thickness.get()))))
        
        ttk.Label(draw_row1, text="点の大きさ:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Scale(draw_row1, from_=1, to=15, variable=self.point_radius,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=5)
        self.point_radius_label = ttk.Label(draw_row1, text="6", width=3)
        self.point_radius_label.pack(side=tk.LEFT)
        self.point_radius.trace_add("write", lambda *a: self.point_radius_label.config(text=str(int(self.point_radius.get()))))
        
        draw_row2 = ttk.Frame(draw_frame)
        draw_row2.pack(fill=tk.X, pady=2)
        
        # 背景色
        ttk.Label(draw_row2, text="背景色:").pack(side=tk.LEFT)
        self.bg_color_combo = ttk.Combobox(draw_row2, values=["black", "white", "green", "blue", "custom"], 
                                            state="readonly", width=10)
        self.bg_color_combo.pack(side=tk.LEFT, padx=5)
        self.bg_color_combo.set("black")
        self.bg_color_combo.bind("<<ComboboxSelected>>", self.on_bg_color_change)
        
        self.bg_color_btn = ttk.Button(draw_row2, text="🎨", width=3, command=self.choose_bg_color)
        self.bg_color_btn.pack(side=tk.LEFT)
        
        # 単色モード
        ttk.Checkbutton(draw_row2, text="単色モード", variable=self.single_color_mode).pack(side=tk.LEFT, padx=(20, 5))
        
        ttk.Label(draw_row2, text="骨格色:").pack(side=tk.LEFT)
        self.pose_color_btn = ttk.Button(draw_row2, text="選択色", command=self.choose_custom_color)
        self.pose_color_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(draw_row2, text="手の色:").pack(side=tk.LEFT, padx=(10, 0))
        self.hand_color_btn = ttk.Button(draw_row2, text="緑", state="disabled")
        self.hand_color_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(draw_row2, text="顔の色:").pack(side=tk.LEFT, padx=(10, 0))
        self.face_color_btn = ttk.Button(draw_row2, text="黄", state="disabled")
        self.face_color_btn.pack(side=tk.LEFT, padx=5)
        
        # 骨格を描出ボタン（F5）
        ttk.Button(control_frame, text="🎨 骨格を描出 (F5)", command=self.process_image).pack(pady=5)
        self.root.bind('<F5>', lambda e: self.process_image())
        
        # プレビューエリア
        preview_frame = ttk.LabelFrame(parent, text="プレビュー", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # オーバーレイ表示切替
        ttk.Checkbutton(preview_frame, text="オーバーレイ表示", variable=self.overlay_display,
                        command=self.update_preview_display).pack(anchor=tk.W)
        
        # キャンバスフレーム
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 入力画像キャンバス
        input_canvas_frame = ttk.Frame(canvas_frame)
        input_canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(input_canvas_frame, text="🖼️ 入力画像").pack()
        self.input_canvas = tk.Canvas(input_canvas_frame, bg='gray20')
        self.input_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 骨格画像キャンバス
        pose_canvas_frame = ttk.Frame(canvas_frame)
        pose_canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.pose_canvas_label = ttk.Label(pose_canvas_frame, text="🦴 骨格フレーム")
        self.pose_canvas_label.pack()
        self.pose_canvas = tk.Canvas(pose_canvas_frame, bg='gray20')
        self.pose_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 処理ログ
        log_frame = ttk.LabelFrame(parent, text="処理ログ", padding="5")
        log_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD)
        self.log_text.pack(fill=tk.X)
    
    def update_complexity_label(self, *args):
        self.complexity_label.config(text=str(self.complexity.get()))
    
    def update_visibility_from_slider(self, *args):
        self.visibility_entry.delete(0, tk.END)
        self.visibility_entry.insert(0, f"{self.visibility.get():.2f}")
    
    def on_visibility_entry(self, event):
        try:
            value = float(self.visibility_entry.get())
            if 0 <= value <= 1:
                self.visibility.set(value)
        except ValueError:
            pass
    
    def on_bg_color_change(self, event):
        color_map = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0)
        }
        selected = self.bg_color_combo.get()
        if selected in color_map:
            self.background_color = color_map[selected]
        elif selected == "custom":
            self.choose_bg_color()
    
    def choose_bg_color(self):
        color = colorchooser.askcolor(initialcolor=self.background_color)
        if color[0]:
            self.background_color = tuple(map(int, color[0][::-1]))  # RGB to BGR
            self.bg_color_combo.set("custom")
    
    def choose_custom_color(self):
        color = colorchooser.askcolor(initialcolor=self.custom_color)
        if color[0]:
            self.custom_color = tuple(map(int, color[0][::-1]))  # RGB to BGR
    
    def browse_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[("画像ファイル", "*.jpg *.jpeg *.png *.gif *.bmp *.webp"), ("すべてのファイル", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            self.batch_files = []
            self.batch_label.config(text="")
            self.show_image_preview(filename)
    
    def browse_multiple_images(self):
        filenames = filedialog.askopenfilenames(
            filetypes=[("画像ファイル", "*.jpg *.jpeg *.png *.gif *.bmp *.webp"), ("すべてのファイル", "*.*")]
        )
        if filenames:
            self.batch_files = list(filenames)
            self.input_file.set(filenames[0])
            self.batch_label.config(text=f"バッチ処理: {len(filenames)}ファイル選択中")
            self.show_image_preview(filenames[0])
    
    def browse_image_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(folder).glob(f"*{ext}"))
                image_files.extend(Path(folder).glob(f"*{ext.upper()}"))
            
            if image_files:
                self.batch_files = [str(f) for f in image_files]
                self.input_file.set(str(image_files[0]))
                self.batch_label.config(text=f"バッチ処理: {len(image_files)}ファイル選択中")
                self.show_image_preview(str(image_files[0]))
            else:
                messagebox.showwarning("警告", "フォルダ内に画像ファイルがありません")
    
    def browse_output_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir.set(folder)
    
    def show_image_preview(self, image_path):
        """画像プレビューを表示"""
        try:
            img = Image.open(image_path)
            # キャンバスサイズに合わせてリサイズ
            canvas_width = self.input_canvas.winfo_width()
            canvas_height = self.input_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            else:
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            self.input_photo = ImageTk.PhotoImage(img)
            self.input_canvas.delete("all")
            self.input_canvas.create_image(
                self.input_canvas.winfo_width()//2 if self.input_canvas.winfo_width() > 1 else 200,
                self.input_canvas.winfo_height()//2 if self.input_canvas.winfo_height() > 1 else 200,
                image=self.input_photo
            )
        except Exception as e:
            self.log_message(f"プレビューエラー: {str(e)}")
    
    def update_preview_display(self):
        """プレビュー表示の切り替え"""
        if self.overlay_display.get():
            self.pose_canvas_label.config(text="🎭 オーバーレイ表示")
        else:
            self.pose_canvas_label.config(text="🦴 骨格フレーム")
    
    def log_message(self, message):
        """ログメッセージを追加"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def process_image(self):
        """画像処理を実行"""
        if self.is_processing:
            messagebox.showwarning("警告", "処理中です。しばらくお待ちください。")
            return
        
        if not self.input_file.get() and not self.batch_files:
            messagebox.showerror("エラー", "画像ファイルを選択してください")
            return
        
        # 処理をスレッドで実行
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_image_thread)
        self.processing_thread.start()
    
    def _process_image_thread(self):
        """画像処理のスレッド処理"""
        try:
            files_to_process = self.batch_files if self.batch_files else [self.input_file.get()]
            output_dir = self.output_dir.get()
            
            self.log_message(f"\n{'='*50}")
            self.log_message(f"処理開始: {len(files_to_process)}ファイル")
            
            start_time = time.time()
            success_count = 0
            
            for i, file_path in enumerate(files_to_process, 1):
                self.log_message(f"\n[{i}/{len(files_to_process)}] 処理中: {Path(file_path).name}")
                
                success = process_single_image(
                    file_path,
                    output_dir,
                    self.mode.get(),
                    self.complexity.get(),
                    self.visibility.get(),
                    self.line_thickness.get(),
                    self.point_radius.get(),
                    self.background_color,
                    self.single_color_mode.get(),
                    self.custom_color,
                    self.single_color_mode.get(),
                    self.log_message
                )
                
                if success:
                    success_count += 1
                    # 最初のファイルの結果をプレビューに表示
                    if i == 1:
                        self.show_result_preview(file_path, output_dir)
            
            elapsed = time.time() - start_time
            self.log_message(f"\n{'='*50}")
            self.log_message(f"✅ 処理完了: {success_count}/{len(files_to_process)}ファイル成功")
            self.log_message(f"処理時間: {elapsed:.1f}秒")
            self.log_message(f"保存先: {output_dir}")
            
            messagebox.showinfo("完了", 
                                f"画像処理が完了しました!\n\n"
                                f"成功: {success_count}/{len(files_to_process)}ファイル\n"
                                f"処理時間: {elapsed:.1f}秒\n\n"
                                f"保存先:\n{output_dir}")
        
        except Exception as e:
            self.log_message(f"❌ エラー: {str(e)}")
            messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{str(e)}")
        
        finally:
            self.is_processing = False
    
    def show_result_preview(self, input_path, output_dir):
        """処理結果をプレビューに表示"""
        try:
            base_name = Path(input_path).stem
            
            if self.overlay_display.get():
                result_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            else:
                result_path = os.path.join(output_dir, f"{base_name}_pose.png")
            
            if os.path.exists(result_path):
                img = Image.open(result_path)
                canvas_width = self.pose_canvas.winfo_width()
                canvas_height = self.pose_canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                else:
                    img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                self.pose_photo = ImageTk.PhotoImage(img)
                self.pose_canvas.delete("all")
                self.pose_canvas.create_image(
                    self.pose_canvas.winfo_width()//2 if self.pose_canvas.winfo_width() > 1 else 200,
                    self.pose_canvas.winfo_height()//2 if self.pose_canvas.winfo_height() > 1 else 200,
                    image=self.pose_photo
                )
        except Exception as e:
            self.log_message(f"結果プレビューエラー: {str(e)}")

# ----------------------------------------------------------------------
# メイン起動
# ----------------------------------------------------------------------
def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    root.geometry("1200x800")
    app = PoseExtractorAppFree(root)
    
    # Ctrl+Q で終了
    root.bind('<Control-q>', lambda e: root.quit())
    
    root.mainloop()

if __name__ == "__main__":
    main()
