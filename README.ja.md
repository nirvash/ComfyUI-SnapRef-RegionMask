# ComfyUI-SnapRef-RegionMask

[English](README.md) | 日本語

**キャラごとに参照画像を1枚ずつ。それぞれの参照は、そのキャラが立つ場所でだけ使われます。**

[SnapRef](https://github.com/ArcAnima/SnapRef) を使うと、Anima で参照画像1枚からキャラクターを描けます。
ところが別々のキャラを2人（`image` と `image_2`）指定すると、両方の参照が両方の人物に混ざります。
片方の髪・眼鏡・目の色・服が、もう片方にも出てしまいます。

このノードは、SnapRef の参照ごとに範囲を割り当てます。参照1は範囲1の中だけ、参照2は範囲2の中だけで使われます。
範囲は、左右または上下の分割、長方形、任意の MASK で指定できます。

![ノードなしとありの比較](docs/comparison.png)

*シードとプロンプトは同じ。上段：SnapRef に参照2枚。左の子に右の子の眼鏡と紫の目が移っている。
下段：＋ SnapRef Region Mask（`layout = left_right`）。*

![ピクニック、ショッピング、ゲームセンター](docs/situations.png)

![ちびキャラの参照：左右と上下](docs/chibi.png)

## 仕組み

SnapRef は、Anima のすべての self-attention に参照画像の key/value トークンを追加します。
このノードは、参照ごとに注意スコアへトークン単位のバイアスを足します。

- 参照の範囲内：`0`（SnapRef とまったく同じように参照を見る）
- 範囲外：`log(outside_strength)`。既定では `-inf`（まったく見ない）
- 長方形の範囲は境界を `feather` の幅でなだらかにする。MASK の場合は、マスク自体のぼかしがそのまま使われる

範囲内の計算は、通常の SnapRef と同じです。テキストのプロンプト（cross-attention）・SnapRef の重み・ComfyUI には手を加えません。
注意計算への差し込みはモデルの呼び出しごとに行い、終わったらすぐ元に戻すので、モデルは SnapRef が設定した状態のまま残ります。

最初の `ref_start` の割合のステップでは、どのトークンも参照を見ません。先にプロンプトで人物の配置を決めるためです。

## 必要なもの

- Anima に対応した ComfyUI
- [SnapRef](https://github.com/ArcAnima/SnapRef) のノードとアダプター（検証には `snapref_v2a_pre2.safetensors` を使用）
- Anima 系の拡散モデル、テキストエンコーダー `qwen_3_06b_base`、VAE `qwen_image_vae`
- fp16/bf16 が使える CUDA GPU（マスク付きの注意計算に、log-sum-exp を返す SDPA カーネルを使います）

## インストール

```
cd ComfyUI/custom_nodes
git clone https://github.com/nirvash/ComfyUI-SnapRef-RegionMask
```

ComfyUI を再起動すると、**SnapRef → SnapRef Region Mask** にノードが表示されます。

## サンプルワークフロー

`examples/input` の画像を `ComfyUI/input` にコピーし、ワークフローを ComfyUI にドラッグ＆ドロップします。
UNET / CLIP / VAE / SnapRef アダプターのファイル名を手元のものに変更して、実行してください。
各 `*_output.png` はそのワークフローの出力で、メタデータにワークフローが入っています。この画像をドラッグしても読み込めます。

| ワークフロー | 範囲 | 参照 | 出力 |
|---|---|---|---|
| `snapref_region_mask_sample_boxes.json` | `layout = left_right`、`split = 0.5` | A（左）、B（右） | `sample_boxes_output.png` |
| `snapref_region_mask_sample_masks.json` | `mask_1` ＝上半分、`mask_2` ＝下半分（Load Image (as Mask) で読み込み） | C（上）、D（下） | `sample_masks_output.png` |

`*_api.json` は、同じグラフの API 形式です。
**SnapRef Region Mask** ノードをバイパス（Ctrl+B）すると、ノードなしで参照が混ざる様子を確認できます。

| A | B | C | D |
|---|---|---|---|
| <img src="examples/input/snapref_region_mask_ref_A.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_B.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_C.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_D.png" width="160"> |

## つなぎ方

```
UNETLoader ─┐
CLIPLoader ─┼─ SnapRef（image = A, image_2 = B, multi_layout = separate）
VAELoader  ─┘        │
                SnapRef Region Mask（layout / split、または mask_1 / mask_2）
                     │
                  KSampler ── CLIPTextEncode（「…, the girl on the left …, the girl on the right …」）
```

SnapRef の直後に置いてください。領域プロンプト用のノードは不要です。

## 入力

| 入力 | 既定値 | 意味 |
|---|---|---|
| `model` | | SnapRef の MODEL 出力 |
| `layout` | `left_right` | `left_right`：参照1（`image`）は `split` より左、参照2（`image_2`）は右。`top_bottom`：参照1は上、参照2は下。`boxes`：下の長方形の入力を使う |
| `split` | 0.5 | 2つの範囲の境界の位置。幅（`left_right`）または高さ（`top_bottom`）に対する割合 |
| `feather` | 0.05 | 長方形の範囲の境界をぼかす幅（画像全体に対する割合）。画像の端と重なる辺はぼかさない |
| `outside_strength` | 0 | 範囲外でも参照をどれだけ残すか（0＝まったく見ない） |
| `ref_start` | 0.1 | 参照をまったく使わないステップの割合（先に構図を決める）。30ステップなら 0.1 で1〜3ステップ目 |
| `x`, `y`, `width`, `height` | 0, 0, 0.5, 1 | `layout = boxes` のときの参照1の範囲。画像全体に対する割合 |
| `x_2`, `y_2`, `width_2`, `height_2` | 0.5, 0, 0, 1 | `layout = boxes` のときの参照2の範囲。`width_2 = 0` なら全参照が範囲1を使う |
| `mask_1`, `mask_2` | | 参照1／2の MASK（白い部分で参照を見る）。つなぐと、その参照については `layout` より優先 |

マスクはどのサイズでも、モデルのトークンの格子に合わせて縮小されます。
マスクエディタで描く（Load Image ノードを右クリック →「Open in MaskEditor」）か、Solid Mask／Mask Composite で作れます。
3枚目の参照（`image_3`）は範囲2を使います。

## コツ

- **SnapRef の `strength` は既定値（0）のままにしてください。** ここに載せた結果はすべて 0 です。
- **プロンプトで各キャラの容姿を簡単に補足すると、再現度が上がります。** 参照画像だけでも2人は描き分けられます。
  そこに左右それぞれの髪・目・服（「the girl on the left has … and wears …」）を書き足すと、より参照に近くなります。
  カフェのテストでは、補足なしだと4シード中3シードで片方の服の色が変わり、補足ありでは8枚中0枚でした。
  この服の色の変化は、通常の SnapRef（参照1枚）でも起きるもので、このノードが原因ではありません。
- **誰がどこにいるかを書いてください。** 範囲はプロンプトの構図と合っている必要があります。キャラが範囲の境界をまたいで描かれると、
  範囲外の部分では参照が効かなくなります（境界に顔がかかると崩れることがあります）。
- **`ref_start`。** `0` と `0.1` で、参照の再現度にはっきりした差は見られませんでした。
  違いが出たのは、2人が近くに並んで立つシーンだけです。`0` では4シード中2シードで片方のキャラが2人に増え（2つの設定で同じように再現）、`0.1` では起きませんでした。
  既定値は `0.1` にしています。
- 範囲内の物に参照が乗ることがあります（例：ぬいぐるみにキャラの顔が付く）。
- 抱き合う、前後に重なるなど、人物が大きく重なる構図には対応していません。

## Anima Regional との併用（任意）

[ComfyUI-Anima-Regional](https://github.com/An1X3R/ComfyUI-Anima-Regional) と併用すると、各キャラの動作を範囲ごとに指定できます。注意点：

- Regional のレイアウトとこのノードで、まったく同じ範囲を指定してください。範囲が重なっていると、境界で顔が崩れました。
- Regional は1つのグラフにつき1回だけ適用してください。
- **プロンプトが更新されない問題（Anima Regional 0.4.7）：** モデルを VRAM に載せたまま Regional のプロンプトを変えると、
  Regional が前回のパッチを包み直し、前回のテキストを使い続けます。プロンプトを変えたら、モデルを解放してください（Unload Models、または `POST /free`）。
  これは SnapRef を使わなくても起きます。
- SnapRef は本命とネガティブのバッチを2回の呼び出しに分けますが、`cond_or_uncond` の目印を元のまま渡します。
  そのため Regional がネガティブ側も振り分けていました（`unusable cond_or_uncond markers [1, 0] for batch 1`）。このノードは、その呼び出しの目印を書き直します。
  ただし、実行ごとに1回、SnapRef が参照をエンコードする間だけはこの警告が残ります（Regional がその処理も振り分けるため）。
  これは SnapRef と Regional の組み合わせによるもので、このノードでは直せません。

## 制限

- マスクが効くのは SnapRef の標準設定のときだけです：`fidelity = 1`、`ref_guidance_on = off`、`ref_in_uncond = off`、
  CUDA の fp16/bf16。それ以外では、SnapRef がそのまま動きます（マスクなし）。
- 検証環境：SnapRef v2a pre2、hakushiMixAnima v1.0（Anima）、30ステップ、er_sde / simple、CFG 4、1216×832 と 832×1216。
- SnapRef の内部（各ブロックの `self_attn` にある `_ref_kv_state`）に依存しています。今後の SnapRef の更新で対応が必要になる可能性があります。

## テスト

```
python -m pytest tests
```

マスク・レイアウト・スケジュール計算・目印の補正・差し込みの復元のテストで、ComfyUI がなくても実行できます。

## ライセンス

Apache License 2.0。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。
SnapRef は Apache-2.0 の別プロジェクトで、アダプターの重みには独自のライセンスがあります。
