# Video Stabilizer

参照映像のルック（輝度・コントラスト・色相・彩度）を、処理対象の動画へ転写する Windows 向けツールです。

開発者向け: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## 使い方（GUI）

1. アプリを起動（`video-stabilizer` または配布 `.exe`）
2. **参照映像** … 合わせたい色味の動画を1本選択
3. **処理対象** … ファイル追加 / フォルダ追加で補正したい動画を登録
4. **出力** … 必要なら出力フォルダ・ファイル名 prefix/suffix・衝突時の動作を設定
5. **実行** … 進捗とログを確認。キャンセルも可能
6. 出力先（未指定時は各入力と同じ場所の `output_corrected/`）に補正済み動画ができます

### プリセット
- **standard** … バランス型（既定）
- **natural** … 弱めの補正
- **strong** … 強めの補正

### CLI（従来方式）
```bash
video-stabilizer --cli
```
参照動画 → フォルダ選択の2段階ダイアログで処理します。

## 準備（開発）
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) を参照してください。

## 配布
Releases から `SBC_VideoStabilizer_vXX.exe` を取得して実行できます。

## 問題報告
Issues へお願いします。
