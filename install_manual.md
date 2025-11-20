# MediaPipe Pose Extractor インストールマニュアル

## 動作環境

- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.8 〜 3.11（3.12以降は MediaPipe 未対応の場合あり）
- **メモリ**: 8GB以上推奨
- **ストレージ**: 2GB以上の空き容量

---

## インストール手順

### 1. Python のインストール

Python がインストールされていない場合は、公式サイトからダウンロードしてインストールしてください。

**ダウンロード**: https://www.python.org/downloads/

インストール時に **「Add Python to PATH」にチェック** を入れてください。

インストール確認：
```bash
python --version
```

### 2. 仮想環境の作成

アプリ専用の仮想環境を作成します。これにより、他のプロジェクトとの依存関係の競合を防ぎます。

#### Windows の場合

```bash
# 作業フォルダを作成して移動
mkdir pose_extractor
cd pose_extractor

# 仮想環境を作成
python -m venv venv

# 仮想環境を有効化
venv\Scripts\activate
```

#### macOS / Linux の場合

```bash
# 作業フォルダを作成して移動
mkdir pose_extractor
cd pose_extractor

# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate
```

仮想環境が有効化されると、コマンドプロンプトの先頭に `(venv)` と表示されます。

### 3. 必要なモジュールのインストール

仮想環境を有効化した状態で、以下のコマンドを実行してください。

```bash
# pip を最新版に更新
pip install --upgrade pip

# 必須モジュールをインストール
pip install opencv-python
pip install mediapipe
pip install numpy
pip install Pillow

# ドラッグ＆ドロップ機能用（オプション）
pip install tkinterdnd2
```

#### 一括インストール（requirements.txt を使用）

以下の内容で `requirements.txt` を作成し、一括インストールすることもできます。

**requirements.txt**
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
Pillow>=10.0.0
tkinterdnd2>=0.4.3
```

```bash
pip install -r requirements.txt
```

### 4. アプリファイルの配置

ダウンロードした `pose_extractor_tk_v3.py` を作業フォルダに配置します。

```
pose_extractor/
├── venv/
├── pose_extractor_tk_v3.py
└── requirements.txt（任意）
```

### 5. 動作確認

```bash
# 仮想環境が有効化されていることを確認
# (venv) が表示されていればOK

# アプリを起動
python pose_extractor_tk_v3.py
```

アプリのウィンドウが表示されれば、インストール完了です。

---

## トラブルシューティング

### 「ModuleNotFoundError: No module named 'cv2'」

OpenCV がインストールされていません。
```bash
pip install opencv-python
```

### 「ModuleNotFoundError: No module named 'mediapipe'」

MediaPipe がインストールされていません。
```bash
pip install mediapipe
```

### 「tkinterdnd2 がインストールできない」

tkinterdnd2 はドラッグ＆ドロップ機能に必要ですが、なくてもアプリは動作します。
インストールできない場合は、ファイル選択ボタンから画像/動画を選択してください。

### 「Python 3.12 で MediaPipe がインストールできない」

MediaPipe は Python 3.12 以降に未対応の場合があります。
Python 3.10 または 3.11 を使用してください。

### Windows で「python が見つからない」

Python のパスが通っていません。以下を試してください：
- `python` の代わりに `py` を使用
- Python を再インストールし「Add Python to PATH」にチェック

### 仮想環境の再有効化

ターミナルを閉じた後、再度アプリを起動する場合は仮想環境を有効化してください。

**Windows**
```bash
cd pose_extractor
venv\Scripts\activate
python pose_extractor_tk_v3.py
```

**macOS / Linux**
```bash
cd pose_extractor
source venv/bin/activate
python pose_extractor_tk_v3.py
```

---

## アンインストール

仮想環境ごとフォルダを削除するだけでアンインストールできます。

```bash
# pose_extractor フォルダを削除
```

システムの Python 環境には影響しません。

---

## 次のステップ

インストールが完了したら、**操作ガイドマニュアル**を参照してアプリの使い方を確認してください。
