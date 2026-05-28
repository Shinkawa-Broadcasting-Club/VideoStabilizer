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
### GUI（既定）
```bash
uv run video-stabilizer
# または
uv run python main.py
```

### CLI（従来のファイルダイアログ）
```bash
uv run video-stabilizer --cli
uv run python main.py --cli
```

## 4. テスト
```bash
uv run python -m unittest discover -s tests -v
```

受け入れ基準（KPI）: [ACCEPTANCE.md](ACCEPTANCE.md)

## 5. ビルド

```powershell
pyinstaller --onefile `
  --name SBC_VideoStabilizer_dev `
  --collect-binaries av `
  --collect-binaries cv2 `
  --collect-submodules numexpr `
  --exclude-module tkinter `
  --windowed `
  main.py
```

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
- `VS_COLLISION_POLICY` (`overwrite` / `skip` / `rename`)
- `VS_OUTPUT_DIR`（統一出力フォルダ）
- `VS_USE_GUI_PROGRESS`（GUI モードでは自動で true）

## 7. UI 設定の保存場所
- Windows: `%APPDATA%\VideoStabilizer\settings.json`
- その他: `~/.video_stabilizer/settings.json`

## 8. バッチ manifest
出力フォルダ（または入力フォルダ）に `manifest.json` が生成されます。再実行時の「完了済みスキップ」「失敗のみ再試行」に使用します。
