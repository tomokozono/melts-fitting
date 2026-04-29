# melts-fitting

MELTS シミュレーション出力 (`melts-liquid.tbl`) から、火山岩マグマの物性値を関数フィッティングするツール群。

## 開発の経緯

MELTS で計算した液相ライン (P-T-組成) データに対して、マグマ上昇モデルなどで利用できる経験式を当てはめる必要が生じたことが発端。当初は Excel ファイルを直接読み込む形だったが、再現性・移植性のために CSV 形式の `melts-liquid.tbl` を直接入力とする CLI スクリプトに整理した。

フィッティング対象と採用モデルは以下のとおり。

| 対象 | モデル | 参照 |
|------|--------|------|
| H2O 溶解度 vs 圧力 | べき乗則 `H2O = a · P^b` | Burnham & Davis (1974) |
| 粘性 vs H2O 含有量 | Giordano (2008) VTF + 3次多項式フィット / H&D-form 6パラメータフィット | Giordano et al. (2008); Hess & Dingwell (1996) |
| 結晶化度 vs 圧力 | 指数モデル or 区分的モデル (AIC で自動選択) | — |

粘性フィッティングでは、全組成が圧力とともに変化するケース (full composition) と、基準組成を固定して H2O だけを変化させるケース (fixed composition, Index=1 の行を参照) の両方を計算する。結晶化度フィッティングでは、指数モデルと区分的モデル (高圧側: 指数関数, 低圧側: 2次多項式、接続点 BP を自由パラメータとして最適化) を AIC で比較し、より良いモデルを自動選択する。

## ファイル構成

```
melts-fitting/
├── fit_h2o.py            # H2O 溶解度フィッティング
├── fit_viscosity.py      # 粘性フィッティング
├── fit_crystallinity.py  # 結晶化度フィッティング
├── run_fitting.sh        # 上記3スクリプトを一括実行するシェルスクリプト
├── pyproject.toml        # 依存関係定義 (uv)
└── melts-liquid.tbl      # MELTS 出力テーブル (例)
```

## 入力データ形式

MELTS が出力する `melts-liquid.tbl` (CSV 形式) を使用する。必要な列は以下のとおり。

| 列名 | 説明 |
|------|------|
| `Index` | ステップ番号 (1 始まり) |
| `P (kbars)` | 圧力 (kbar) |
| `T (C)` | 温度 (°C) |
| `wt% H2O` | H2O 含有量 (wt%) |
| `wt% SiO2` 〜 `wt% P2O5` | 各酸化物の wt% |
| `liq V (cc)` | 液相体積 (cc) |
| `liq vis (log 10 poise)` | MELTS 計算粘性 (log10 poise) |

## セットアップ

[uv](https://github.com/astral-sh/uv) を使用。

```bash
uv sync
```

## 実行方法

### 一括実行 (推奨)

```bash
./run_fitting.sh melts-liquid.tbl
```

別ディレクトリのファイルを指定した場合、出力もそのディレクトリに保存される。

```bash
./run_fitting.sh path/to/data/melts-liquid.tbl
```

### 個別実行

```bash
uv run python fit_h2o.py          melts-liquid.tbl
uv run python fit_viscosity.py    melts-liquid.tbl
uv run python fit_crystallinity.py melts-liquid.tbl
```

## 出力ファイル

各スクリプトは入力ファイルと同じディレクトリに以下を出力する。

| ファイル | 内容 |
|----------|------|
| `h2o_fitting.png` | H2O フィッティング曲線と残差プロット |
| `h2o_fitting_coeffs.txt` | H2O フィッティング係数 (a, b, R², N, P範囲) |
| `viscosity_giordano_poly3.png` | 粘性フィッティング曲線と残差プロット |
| `viscosity_fitting_coeffs.txt` | 粘性フィッティング係数 (多項式 / H&D-form) |
| `viscosity_scatter_data.csv` | 粘性フィッティングの散布図データ |
| `viscosity_curve_data.csv` | 粘性フィッティングの曲線データ |
| `crystallinity_vs_P.png` | 結晶化度フィッティング曲線と残差プロット |
| `crystallinity_fitting_coeffs.txt` | 結晶化度フィッティング係数 (両モデル + 選択結果) |

## フィッティングモデル詳細

### H2O 溶解度

```
H2O (fraction) = a · P (Pa) ^ b
```

参照式 (Burnham & Davis 1974): `H2O = 4.11×10⁻⁶ · P^0.5`

### 粘性

**3次多項式フィット**
```
log10(η [Pa·s]) = a0 + a1·X + a2·X² + a3·X³,  X = H2O (fraction)
```

**H&D-form フィット (Hess & Dingwell 形式, 6パラメータ)**
```
log10(η) = a + b·x + (c + d·x) / (T_K − (e + f·x)),  x = ln(wt% H2O)
```

### 結晶化度

**Model A — 指数モデル**
```
C = A · exp(−k · P [MPa])
```

**Model B — 区分的モデル (接続点 BP を自由パラメータとして最適化)**
```
P ≥ BP:  C = A · exp(−k · P)
P < BP:  C = a3·(P − BP)² + b3·(P − BP) + v2   (v2 = A·exp(−k·BP), 連続性保証)
```

AIC が小さいモデルを自動選択する。

## 依存ライブラリ

- numpy
- scipy
- pandas
- matplotlib
- openpyxl
