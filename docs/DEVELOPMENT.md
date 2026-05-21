# DEVELOPMENT

## 1. 開発環境
- Python 3.12
- Windows 11 24H2

## 2. 環境構築
### 2.1 `uv` を使う（推奨）
```bash
uv sync
```
### 2.2 `pip + venv` を使う
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## 3. 起動
```bash
# uv使用時
uv run video-stabilizer
```
```bash
# pip+venv使用時
python main.py
```

## 4. テスト
### 4.1 ユニットテスト

```bash
# uv使用時
uv run python -m unittest discover -s tests -v
```
```bash
# pip+venv使用時
python -m unittest discover -s tests -v
```

## 5. ビルド

GitHub Actionsで実行しているCIと同じような動作をします

```powershell
pyinstaller --onefile `
  --name SBC_VideoStabilizer_dev `
  --collect-binaries av `
  --collect-binaries cv2 `
  --collect-submodules numexpr `
  --hidden-import tkinter `
  --hidden-import tkinter.filedialog `
  main.py
```
生成物は `dist/` 配下に出力されます。

## 6. 主要な環境変数
### ログ
- `VS_LOG_LEVEL`（例: `DEBUG`, `INFO`）

### 処理パラメータ（`video_stabilizer/config.py`）
- `VS_TARGET_SAMPLING_SEC`
- `VS_EMA_ALPHA`
- `VS_CHROMA_SOFT_CLIP_THRESHOLD`
- `VS_CHROMA_SOFT_CLIP_DIFF`
- `VS_RATIO_CLIP_LOW`
- `VS_RATIO_CLIP_HIGH`
- `VS_STATS_MAX_DIM`
- `VS_QUEUE_MULTIPLIER`
- `VS_ON_FRAME_FAILURE` (`hold` / `black` / `abort`)
- `VS_MITCHELL_T_MAX`
- `VS_PRESERVE_AUDIO` (`true` / `false`, 既定: `true`)
  - `true`: 補正済み映像と元ファイルの音声を、最終拡張子のコンテナへ remux
  - `false`: 音声なしで最終拡張子のコンテナへ remux
  