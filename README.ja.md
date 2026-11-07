# ComfyUI-SnapRef-RegionMask

[English](README.md) | 日本語

Animaの[SnapRef](https://github.com/ArcAnima/SnapRef)に、参照画像ごとの領域を指定するノードです。2人の髪・服・小物の混入を抑えます。左右・上下の分割、長方形、任意のMASKが使えます。領域を描かずに、生成そのものから推定することもできます（[自動マスク](#自動マスク試験的)、試験的）。

![上段：SnapRefのみ、下段：Region Maskあり。同じシードとプロンプト](docs/comparison.png)

![ピクニック・ショッピング・クレーンゲーム](docs/situations.png)

![ちびキャラの左右・上下配置](docs/chibi.png)

![自動マスク。上：参照C・D、下：最終画像に重ねた推定領域と最終画像](docs/auto_mask.png)

## 導入・使い方

Anima対応のComfyUI、SnapRefノードとアダプター、Animaのモデル・テキストエンコーダー・VAE、fp16/bf16対応のCUDA GPUが必要です。自動マスクの最小カットには `pip install pymaxflow` が必要です（無い場合は Auto Mask が明確なエラーを出します）。

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/nirvash/ComfyUI-SnapRef-RegionMask
```

ComfyUIを再起動し、SnapRefのMODEL出力とKSamplerの間に **SnapRef → SnapRef Region Mask** を接続します。SnapRefは `multi_layout = separate`、キャラごとに参照1枚を指定してください。領域プロンプト用ノードは不要です。

## サンプル

[生成画像とプロンプトの内容一覧](examples/README.md)

`examples/input` の画像を `ComfyUI/input` にコピーし、JSONをComfyUIで開きます。モデル・CLIP・VAE・アダプターのファイル名を手元のものに変更して実行してください。

| ワークフロー（examples内） | 内容 | 出力 |
|---|---|---|
| `snapref_region_mask_sample_boxes.json` | Aを左、Bを右に配置 | `sample_boxes_output.png` |
| `snapref_region_mask_sample_masks.json` | MASKでCを上、Dを下に配置 | `sample_masks_output.png` |
| `snapref_region_mask_sample_arcade.json` | 参照を左右交換：Bが左で△口、Aが右側の筐体を操作。boxで筐体を参照領域から除外 | `sample_arcade_output.png` |
| `snapref_region_mask_sample_auto.json` | ピクニック：Cが左、Dが右。領域を描かずに自動マスクで生成（人物語 `girl on the left` / `girl on the right`） | `sample_auto_output.png`（推定領域：`sample_auto_regions.png`） |

`*_api.json` はAPI形式です。出力PNGにもワークフローを埋め込んでいます。Region Maskノードをバイパス（Ctrl+B）すると、同条件でノードなしの結果を比較できます。

| A | B | C | D |
|---|---|---|---|
| <img src="examples/input/snapref_region_mask_ref_A.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_B.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_C.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_D.png" width="160"> |

## 入力

数値範囲はノードのUI設定値です。座標・寸法は画像全体を1とする割合です。

| 入力 | 範囲・選択肢 | 既定値 | 説明 |
|---|---|---|---|
| `model` | MODEL | — | SnapRefのMODEL出力 |
| `layout` | `left_right` / `top_bottom` / `boxes` | `left_right` | 左右分割・上下分割・長方形指定。参照1は左／上、参照2は右／下 |
| `split` | 0.05–0.95 | 0.5 | 左右／上下分割の境界 |
| `feather` | 0–0.5 | 0.05 | 長方形の境界をぼかす幅。MASKは元の濃淡を使用 |
| `outside_strength` | 0–1 | 0 | 領域外で参照を残す強さ。0は無効、1は領域外も同じ強さ |
| `ref_start` | 0–0.95 | 0.1 | 最初に参照を使わないステップの割合 |
| `x`, `y` | 各0–1 | 0, 0 | `boxes`の参照1の左上座標 |
| `width`, `height` | 各0–1 | 0.5, 1 | 参照1の幅・高さ |
| `x_2`, `y_2` | 各0–1 | 0.5, 0 | 参照2の左上座標 |
| `width_2`, `height_2` | 各0–1 | 0, 1 | 参照2の幅・高さ。`width_2 = 0`は全参照に領域1を使用 |
| `mask_1`, `mask_2` | MASK、値0–1 | 未接続 | 白い部分で参照を見る。その参照のレイアウト指定より優先 |

MASKはモデルのトークン格子に合わせてリサイズされます。3枚目の参照（`image_3`）は領域2を使用します。

人物が画面の一部に寄る場面では、`boxes`で人物の範囲だけを指定し、筐体などを参照領域から外せます。

## 自動マスク（試験的）

構図を指定しないプロンプト向けに、領域を手で描く代わりに、生成そのものから人物ごとの領域を推定します。

| ノード | 役割 |
|---|---|
| SnapRef Region Probe | MODEL → MODEL, REGION_PROBE。画像トークンがプロンプト中の2人の人物語にどう注目したかを記録します（`person_1` は SnapRef の `image`、`person_2` は `image_2` に対応。言い換えは `\|` 区切り）。`auto_mask` に重みの名前を指定し、モデルの流れの最後に置きます。`instance_mask`（既定オン）で、ブロック16で画像トークンどうしがどう参照し合っているかも1 step 分記録し、人物をまとまりとして扱えるようにします。`ref_affinity` をオンにすると（SnapRef Region Mask の後ろ）、各画素がどちらの参照に似ているかも記録します |
| SnapRef Region Auto Mask | IMAGE + REGION_PROBE → SnapRef Region Mask 用の `mask_1`、`mask_2`。記録した自己注意で画像トークンをまとまりに分け、まとまりごとに注目している人物語の人物を割り当て、SLIC で輪郭を画像の境目に合わせ、superpixel のグラフ上で最小カット（既定。`pymaxflow` が必要）で弱い手がかりを隣側へまとめる。29枚のテストセットで最悪場面の取り違え 11.7%→10.0%（全場面10%以下）、追加約5ms。`segmentation_method = existing` で従来の多数決。GPU で約0.2秒。`source` は `attention`（人物語）、`reference`（参照への類似度）、`both`（人物語に加え、帽子などを類似度で足す。`ref_weight` 1） |
| SnapRef Region Auto Mask Panel | 推定に使った画像と最終画像に領域を重ねた確認用の画像を作ります |

推奨構成（2つの段で seed・sampler・scheduler・CFG を揃え、30 step）:

1. **推定**: モデル → Region Probe（`auto_mask`、`instance_mask` オン）→ KSamplerAdvanced で 0〜8 step（`return_with_leftover_noise = disable`）→ VAEDecode → Auto Mask → マスク
2. **本番**: SnapRef → Region Mask（そのマスク、`ref_start = 0.1`）→ KSampler で 30 step

マスクは参照なしの 8 step 目から作るので、誰がどこにいるかはプロンプトで決める必要があります。人物語は構図に合った語にし、プロンプトでも同じ語で人物を指して、見た目をその人物に結び付けて書きます。

- 左右に並ぶ: `girl on the left` / `girl on the right`（「the girl on the left has … and wears …」）
- 膝枕: `lying girl` / `sitting girl`（「the sitting girl sits on the couch, the lying girl rests her head on the sitting girl's lap, the sitting girl has …」）。横になる子の頭は反対側に来るので、左右の語は合いません
- 女の子と男性: `1girl | girl` / `1man | man`

膝枕3枚（本番で参照を使用、正解は手で輪郭を引いたもの）で、人物の画素のうち相手側になったのは3.8%でした。1216×832 で1枚約16秒です。見た目を詳しく書くほど、参照よりプロンプトに引きずられるので、参照で目立つ特徴にとどめてください。

SnapRef で構図が大きく動く場合は、3段にする方法もあります。4 step の下書き → そのマスクで SnapRef ありの 0〜8 step を描いて推定し直す（`source = both`、8 step の区間の中なので Region Mask の `ref_start = 0.3125`）→ 本番、です。サンプルのワークフロー `examples/snapref_region_mask_sample_auto.json` はこの構成です。膝枕では 9.6% と悪く、約4秒遅くなりました。Region Probe の `text` はポジティブプロンプトと完全に同じにしてください。プロンプトを変えたら、Probe にも同じ文を貼り付けます。

制約:

- 同梱の重み `hakushi_anima_gt_v1` は、hakushiMixAnima v1.0 で人物語 `1girl | girl` と `1man | man` を使って学習しました（手で確認した29枚）。それ以外の人物語（位置・役割の語）では、そのほうが良かったので、Auto Mask は全層の単純平均を使います。髪や服の語を人物語にすると効かず、人物1にはその語の部分しか付きません。
- 2人までです。マスクが制限するのは各参照を読む範囲だけで、プロンプトの語や画像トークンどうしの自己注意は両方の人物に届きます。

[自動マスクの仕組みと、重みの学習方法](docs/auto_mask.ja.md)

## 使用上の注意

- 人物の位置と髪・目・服をプロンプトで簡潔に補足してください。SnapRefの `strength` は既定値の0で検証しています。
- マスクが有効なのは `fidelity = 1`、`ref_guidance_on = off`、`ref_in_uncond = off`、CUDAのfp16/bf16です。それ以外ではマスクなしのSnapRefとして動作します。
- 参照の完全な再現は保証しません。人物が境界をまたぐ構図、大きく重なる構図では混入や崩れが起こり得ます。領域内の小物にも参照が乗る場合があります。
- 検証：SnapRef v2a pre2、hakushiMixAnima v1.0、30 steps、er_sde / simple、CFG 4、1216×832／832×1216。

[仕組み・検証メモ・Anima Regionalとの併用](docs/notes.ja.md)

## テスト・ライセンス

`python -m pytest tests`（ComfyUIなしで実行可能）。Apache-2.0：[LICENSE](LICENSE)、[NOTICE](NOTICE)。SnapRefとアダプターのライセンスはそれぞれの配布元を確認してください。
