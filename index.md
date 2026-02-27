## 日本木材学会大会（2026、広島）ポスター発表 補足資料

本ページでは、学会ポスターで紹介した内容のうち、**主に解析フロー** の詳細を補足します。<br>
ポスター本体では紙面の都合上、省略した数式・再現手順等も含めて整理しています。

また、本ページで紹介するモデルは **PyPI で公開済み** のため、手元の Python 環境でそのまま利用できます。ただし、学習・推論にはGPU の利用を推奨します。GPU 環境が手元にない場合は、Google Colab などのクラウド実行環境を利用してください。

- [公開資料](https://github.com/Mantis-Ryuji/poster-2026)
- [モデル実装](https://github.com/Mantis-Ryuji/ChemoMAE)

---

## 目次

- [日本木材学会大会（2026、広島）ポスター発表 補足資料](#日本木材学会大会2026広島ポスター発表-補足資料)
- [目次](#目次)
- [概要](#概要)
- [データ前処理とデータセット作成](#データ前処理とデータセット作成)
  - [1. 反射率変換](#1-反射率変換)
  - [2. ノルム画像の作成](#2-ノルム画像の作成)
  - [3. 大津の二値化による木材領域マスクの作成](#3-大津の二値化による木材領域マスクの作成)
  - [4. データセット作成](#4-データセット作成)
- [SNV処理の特性](#snv処理の特性)
- [Masked Autoencoder](#masked-autoencoder)
  - [1. パッチ化＋トークン化](#1-パッチ化トークン化)
  - [2. 位置埋め込み](#2-位置埋め込み)
  - [3. 可視部分の抽出](#3-可視部分の抽出)
  - [4. Encoder](#4-encoder)
  - [5. Decoder](#5-decoder)
  - [6. Loss Function](#6-loss-function)
- [教師なしセグメンテーション](#教師なしセグメンテーション)
  - [Cosine K-Means](#cosine-k-means)
- [付録](#付録)

---

## 概要

本手法は、近赤外 (NIR) や可視光等の 1次元スペクトルを対象とした、Masked Autoencoder (MAE) に基づく自己教師あり表現学習である。スペクトルをパッチ単位で部分的にマスクし、欠損情報を含む系列全体を再構成する学習タスクを課すことで、化学組成や物理状態の変遷を内包した潜在表現を獲得する。ラベルを必要としないため、劣化のようなアノテーションコストが困難な現象の特徴抽出に適している。

マスキングと再構成により、モデルは特定波長の局所的な振幅情報への過度な依存を抑制され、スペクトル全体の文脈および波長間の相関構造を学習するよう誘導される。これにより、測定ノイズや局所的ばらつきに由来する非本質的な変動が相対的に棄却され、物理化学的な本質を捉えたロバストな潜在表現が得られる。

さらに、学習される潜在表現は L2 正規化により単位超球面上に拘束する設計である。これは SNV と幾何学的な整合性を確保するためであり、ノルム成分を排除して方向成分（コサイン類似度）へ情報を集約する。 

本手法は単なる次元圧縮に留まらず、様々な下流タスクへ展開するための基盤表現学習として位置づけられる。

<img src="images/workflow.png" width="1000">

<details><summary><b>記号の定義（クリックで展開）</b></summary>

**データ前処理とデータセット作成**

*  $H,W$ ：画像の高さ・幅
*  $C$ ：波長点数（スペクトル次元）
*  $\mathbf{I}\in\mathbb{R}^{H\times W\times C}$ ：観測強度（NIR-HSI）
*  $\mathbf{W}_{\rm ref}\in\mathbb{R}^{1\times W\times C}$ ：白板参照
*  $\mathbf{D}_{\rm ref}\in\mathbb{R}^{1\times W\times C}$ ：暗電流参照
*  $\mathbf{R}\in\mathbb{R}^{H\times W\times C}$ ：反射率
*  $\mathbf{r}_{h,w}\in\mathbb{R}^C$ ：画素 $(h,w)$ のスペクトル（ $\mathbf{R}$ のスペクトル軸ベクトル）
*  $\mathbf{L}\in\mathbb{R}^{H\times W\times C}$ ： $\mathbf{R}$ のノルム画像
*  $l_{h,w}=\|\mathbf{r}_{h,w}\|_2$ ：ノルム画像 $\mathbf{L}$ 画素 $(h,w)$ における画素値
*  $t^*$ ：大津の二値化で得る閾値
*  $\mathbf{B}={b_{h,w}}\in\left\{0,1\right\}^{H\times W}$ ：木材マスク（1=木材, 0=背景）
*  $n\in\left\{1,\dots,N\right\}$ ： サンプル番号（ $N$ ：サンプル総数 ）
*  $\mathcal{D}^{(n)}=\left\{(h,w)\mid b^{(n)}_{h,w}=1\right\}$ ：サンプル $n$ の木材画素集合
*  $N_{\rm pix}=\sum_{n=1}^{N}|\mathcal{D}^{(n)}|$ ：全サンプルでの総画素数（木材領域）
*  $\mathbf{X}_{\rm refl}\in\mathbb{R}^{N_{\rm pix}\times C}$ ：木材画素から抽出して縦連結した反射率スペクトル行列
*  $\tilde{\mathbf{x}}_i\in\mathbb{R}^C$ ： $\mathbf{X}_{\rm refl}$ の第 $i$ 行（反射率スペクトル）
*  $\tilde{\mu}_i,\tilde{\sigma}_i$ ： $\tilde{\mathbf{x}}_i$ の平均・標準偏差
*  $\mathbf{x}_i\in\mathbb{R}^C$ ：SNV後のスペクトル
*  $\mathbf{X}\in\mathbb{R}^{N_{\rm pix}\times C}$ ： $\mathbf{x}_i$ を並べたデータセット行列

**Masked Autoencoder**

*  $P$ ：パッチ数
*  $p=C/P$ ：パッチサイズ（割り切れ前提）
*  $\mathbf{X}^{\rm patch}_i\in\mathbb{R}^{P\times p}$ ：パッチ化スペクトル
*  $d_{\rm model}$ ：トークン埋め込み次元
*  $\mathbf{W}_e,\mathbf{b}_e$ ：トークン化の線形写像パラメータ
*  $\mathbf{T}_i\in\mathbb{R}^{P\times d_{\rm model}}$ ：トークン列（ $\mathbf{T}_i=\mathbf{X}^{\rm patch}_i\mathbf{W}_e+\mathbf{1}\mathbf{b}_e^\top$ ）
*  $\mathbf{t}_{\rm cls}$ ：CLSトークン
*  $\mathbf{E}_{\rm pos}$ ：位置埋め込み
*  $\mathcal{V}\subset\left\{1,\ldots,P\right\}$ ：可視パッチインデックス集合（Encoder入力に残す集合）
*  ${\rm Enc}_\theta$ ：Encoder
*  $\mathbf{h}_{{\rm cls},i}$ ：EncoderのCLS出力
*  $d_z$ ：潜在次元
*  $\mathbf{W}_{\rm proj},\mathbf{b}_{\rm proj}$ ：射影ヘッド
*  $\mathbf{z}_i\in\mathbb{S}^{d_z-1}$ ：L2正規化した潜在（球面）
*  ${\rm Dec}_\phi$ ：Decoder
*  $\hat{\mathbf{x}}_i$ ：再構成スペクトル
*  $\mathcal{M}=\left\{1,\ldots,P\right\}\setminus\mathcal{V}$ ：マスクパッチ集合
*  $\mathbf{m}\in\left\{0,1\right\}^C$ ：波長点レベルのマスク指示子

**Cosine K-Means**

*  $K$ ：クラスタ数
*  $\mathbf{a}=(a_1,\ldots,a_{N_{\rm pix}})^\top\in\{1,\ldots,K\}^{N_{\rm pix}}$ ：ラベルベクトル
*  $\mathbf{Z}\in\mathbb{R}^{N_{\rm pix}\times {d_z}}$ ：特徴量行列
*  $\mathbf{C}\in\mathbb{R}^{K\times d_z}$ ：クラスタ中心行列
*  $\mathbf{S}\in\mathbb{R}^{N_{\rm pix}\times K}$ ：類似度行列

</details><br>

---

## データ前処理とデータセット作成

### 1. 反射率変換

NIR-HSI から得られた試料の観測強度 $\mathbf{I}$ を、白板参照 $\mathbf{W}_{\rm ref}$ および 暗電流参照 $\mathbf{D}_{\rm ref}$ を用いて反射率 $\mathbf{R}$ に変換する：

```math
\mathbf{R}
=
\frac{\mathbf{I}-\mathbf{D}_{\rm ref}}{\mathbf{W}_{\rm ref}-\mathbf{D}_{\rm ref}}
```

$\mathbf{I}$ との演算は要素ごとに行い、必要に応じて $\mathbf{W}_{\rm ref},\mathbf{D}_{\rm ref}$ を高さ方向 $H$ にブロードキャストして用いる。

### 2. ノルム画像の作成

画素 $(h, w)$ における反射率スペクトル $\mathbf{r}_{h, w}$ を、 $\mathbf{R}$ のスペクトル軸方向のベクトルとして定義する：

```math
\mathbf{r}_{h, w}
=
\left[r_{h, w,1}, r_{h, w,2}, \ldots, r_{h, w,C}\right]^\top
\in\mathbb{R}^{C}
```

次に、各画素スペクトルの $\ell_2$ ノルムからノルム画像 $\mathbf{L}$ を構成する：

```math
l_{h, w}
=
\|\mathbf{r}_{h, w}\|_2
=
\sqrt{\mathbf{r}_{h, w}^\top\mathbf{r}_{h, w}}
```

### 3. 大津の二値化による木材領域マスクの作成

ノルム値 $l_{h, w}$ を 2 クラス（背景 / 木材）に分割する閾値 $t^*$ を、大津の二値化により求める。<br>
閾値 $t$ に対して、画素集合を次の 2 クラスに分割する：

```math
\mathcal{C}_0(t)=\{(h, w)\mid l_{h, w}\le t\},\qquad
\mathcal{C}_1(t)=\{(h, w)\mid l_{h, w}> t\}
```

各クラスの画素数を

```math
n_k(t)=|\mathcal{C}_k(t)|\quad (k\in\{0,1\}),\qquad
HW = n_0(t)+n_1(t)
```

とし、各クラス平均および全体平均をそれぞれ

```math
\mu_k(t)=\frac{1}{n_k(t)}\sum_{(h, w)\in\mathcal{C}_k(t)} l_{h, w}\quad (k\in\{0,1\})
```

で定義する。大津の二値化はクラス間分散

```math
\sigma_B^2(t)
=
\frac{n_0(t)}{HW}\frac{n_1(t)}{HW}\left(\mu_0(t)-\mu_1(t)\right)^2
```

を最大化する閾値 $t^*$ を選択する：

```math
t^*=\arg\max_t  \sigma_B^2(t)
```

最後に、二値マスク $\mathbf{B}$ を

```math
b_{h, w}=
\begin{cases}
1 & (l_{h, w}>t^*)\\
0 & (l_{h, w}\le t^*)
\end{cases}
```

として定義する（$1$ を木材領域、 $0$ を背景とする）。<br>
（実務上は $\mathbf{I}$ のノルム画像に対して二値化する方が安定します。）

### 4. データセット作成

サンプル $n\in\left\{1,\dots,N\right\}$ に対して、二値マスク $\mathbf{B}^{(n)}$ により木材領域の画素集合

```math
\mathcal{D}^{(n)}=\{(h,w)\mid b^{(n)}_{h,w}=1\}
```

を定義する。木材領域から反射率スペクトルを抽出して行方向に積み上げた行列を

```math
\mathbf{X}^{(n)}_{\rm refl}
=
\begin{bmatrix}
(\mathbf{r}^{(n)}_{h_1,w_1})^\top\\
(\mathbf{r}^{(n)}_{h_2,w_2})^\top\\
\vdots\\
(\mathbf{r}^{(n)}_{h_{|\mathcal{D}^{(n)}|},w_{|\mathcal{D}^{(n)}|}})^\top
\end{bmatrix}
\in\mathbb{R}^{|\mathcal{D}^{(n)}|\times C}
```

と定義する。ただし ${(h_m,w_m)}_{m=1}^{|\mathcal{D}^{(n)}|}$ は $\mathcal{D}^{(n)}$ の列挙である。<br>
次に、全サンプルについて縦方向に連結することで、反射率スペクトルからなるデータセット行列 $\mathbf{X}_{\rm refl}$ を得る：

```math
\mathbf{X}_{\rm refl}
=
\begin{bmatrix}
\mathbf{X}^{(1)}_{\rm refl}\\
\mathbf{X}^{(2)}_{\rm refl}\\
\vdots\\
\mathbf{X}^{(N)}_{\rm refl}
\end{bmatrix}
\in\mathbb{R}^{N_{\rm pix}\times C},
\quad
N_{\rm pix}=\sum_{n=1}^{N}|\mathcal{D}^{(n)}|
```

最後に、 $\mathbf{X}_{\rm refl}$ の各行ベクトルに対して SNV（Standard Normal Variate）処理を適用し、データセット行列 $\mathbf{X}$ を得る。<br>
ここで、 $\mathbf{X}_{\rm refl}$ の第 $i$ 行ベクトルを $\tilde{\mathbf{x}}_i^\top$ と書き、

```math
\mathbf{X}_{\rm refl}
=
\begin{bmatrix}
\tilde{\mathbf{x}}_1^\top\\
\tilde{\mathbf{x}}_2^\top\\
\vdots\\
\tilde{\mathbf{x}}_{N_{\rm pix}}^\top
\end{bmatrix},
\qquad
\tilde{\mathbf{x}}_i\in\mathbb{R}^{C}
\quad (i=1,\ldots, N_{\rm pix})
```

とする。各 $\tilde{\mathbf{x}}_i$ に対して平均と標準偏差を

```math
\tilde{\mu}_i=\frac{1}{C}\sum_{c=1}^{C} \tilde{x}_{i,c},\qquad
\tilde{\sigma}_i=\sqrt{\frac{1}{C-1}\sum_{c=1}^{C}\left(\tilde{x}_{i,c}-\tilde{\mu}_i\right)^2}
```

と定義し、SNV 後のベクトル $\mathbf{x}_i\in\mathbb{R}^{C}$ を

```math
\mathbf{x}_i
=
\frac{\tilde{\mathbf{x}}_i-\tilde{\mu}_i\mathbf{1}}{\tilde{\sigma}_i}
\in\mathbb{R}^{C},
\qquad
(\mathbf{1}\in\mathbb{R}^{C}:\text{全要素が1のベクトル})
```

で与える。この $\mathbf{x}_i$ を行として並べた行列を $\mathbf{X}$ とする：

```math
\mathbf{X}
=
\begin{bmatrix}
\mathbf{x}_1^\top\\
\mathbf{x}_2^\top\\
\vdots\\
\mathbf{x}_{N_{\rm pix}}^\top
\end{bmatrix}
\in\mathbb{R}^{N_{\rm pix}\times C}
```

---

## SNV処理の特性

SNV 処理は、各スペクトル $\tilde{\mathbf{x}}_i\in\mathbb{R}^C$ に対して平均 $\tilde{\mu}_i$ と標準偏差 $\tilde{\sigma}_i$ を

```math
\tilde{\mu}_i=\frac{1}{C}\sum_{c=1}^{C} \tilde{x}_{i,c},\qquad
\tilde{\sigma}_i=\sqrt{\frac{1}{C-1}\sum_{c=1}^{C}\left(\tilde{x}_{i,c}-\tilde{\mu}_i\right)^2}
```

で定義し、SNV 後のベクトル $\mathbf{x}_i\in\mathbb{R}^C$ を

```math
\mathbf{x}_i
=
\frac{\tilde{\mathbf{x}}_i-\tilde{\mu}_i\mathbf{1}}{\tilde{\sigma}_i}
```

で与える。まず、

```math
\tilde{\sigma}_i=\sqrt{\frac{1}{C-1}\sum_{c=1}^{C}\left(\tilde{x}_{i,c}-\tilde{\mu}_i\right)^2}
```

の両辺を二乗すると、

```math
\tilde{\sigma}_i^2
=
\frac{1}{C-1}\sum_{c=1}^{C}\left(\tilde{x}_{i,c}-\tilde{\mu}_i\right)^2
```

したがって両辺に $(C-1)$ を掛けて、

```math
\sum_{c=1}^{C}\left(\tilde{x}_{i,c}-\tilde{\mu}_i\right)^2
=
(C-1)\tilde{\sigma}_i^2
```

ここで、 $\tilde{\sigma}_i>0$ を仮定して（SNV が定義可能であるため）、両辺を $\tilde{\sigma}_i^2$ で割ると、

```math
\sum_{c=1}^{C}\left(\frac{\tilde{x}_{i,c}-\tilde{\mu}_i}{\tilde{\sigma}_i}\right)^2
=
C-1
\qquad \cdots \text{(1)}
```

次に、 $\mathbf{x}_i$ の各成分は

```math
x_{i,c}=\frac{\tilde{x}_{i,c}-\tilde{\mu}_i}{\tilde{\sigma}_i}
```

であるから、 $\mathbf{x}_i$ の $\ell_2$ ノルムは

```math
\|\mathbf{x}_i\|_2
=
\sqrt{\sum_{c=1}^{C}x_{i,c}^2}
=
\sqrt{\sum_{c=1}^{C}\left(\frac{\tilde{x}_{i,c}-\tilde{\mu}_i}{\tilde{\sigma}_i}\right)^2}
```

よって、(1) を用いて

```math
\boxed{\|\mathbf{x}_i\|_2=\sqrt{C-1}}
```

SNV処理により各スペクトル $\mathbf{x}_i$ の $\ell_2$ ノルムは $\sqrt{C-1}$ に正規化される。したがって、全データは半径 $\sqrt{C-1}$ の超球面

```math
\{\mathbf{x}\in\mathbb{R}^C \mid \|\mathbf{x}\|_2=\sqrt{C-1}\}
=
\sqrt{C-1} \mathbb{S}^{C-1}
```

上に配置される。<br>
本研究ではこの球面構造を尊重し、 **球面幾何** に整合した解析フローを構築した。

---

## Masked Autoencoder

自己教師あり学習は、ラベルなしの大規模データから汎用的表現を獲得する潮流であり、その代表が「一部を隠して復元させる」Masked Autoencoder（MAE）である。源流はNLPのBERTで、入力列のマスク位置が明示された状態で周辺文脈から欠損トークンを推定させることで、トークン間依存を学習し下流タスクに転用可能な表現を得る。これが画像へ拡張されViT-MAEでは、マスク部分をエンコーダに入れず可視パッチのみを処理する非対称エンコーダ・デコーダ構造を採用し、高マスク率でも計算効率と頑健性を両立した。

この系譜は一次元信号であるスペクトル解析にも有効で、局所的吸収帯から広帯域ベースライン変動まで多スケール構造が重畳するスペクトルに対し、波長帯域をマスクして復元させることで、帯域間相関や連続性、形状差分といった本質特徴の学習が促される。

<img src="images/maskedautoencoder.png" width="1000">

### 1. パッチ化＋トークン化

スペクトル $\mathbf{x}_i\in\mathbb{R}^{C}$ をパッチ数 $P$ で分割し、パッチサイズを

```math
p = \frac{C}{P} \quad (\text{割り切れる前提})
```

とする。パッチ行列を

```math
\mathbf{X}^{\rm patch}_i \in \mathbb{R}^{P \times p}
```

と定義する。<br>
トークン化（線形写像）：

```math
\mathbf{T}_i = \mathbf{X}^{\rm patch}_i \mathbf{W}_e + \mathbf{1} \mathbf{b}_e^\top
\in \mathbb{R}^{P \times d_{\rm model}},
\quad
\mathbf{W}_e\in\mathbb{R}^{p \times d_{\rm model}},
\mathbf{b}_e\in\mathbb{R}^{d_{\rm model}}
```

これは各パッチ（長さ $p$ のスペクトル断片）を、Transformer が内部表現として扱う埋め込み次元 $d_{\rm model}$ へ写像するための線形埋め込みである。

### 2. 位置埋め込み

CLS トークンを $\mathbf{t}_{\rm cls}\in\mathbb{R}^{d_{\rm model}}$ とし、フル系列（CLS + 全パッチ）を
```math
\mathbf{U}^{\rm full}_i
=
\begin{bmatrix}
\mathbf{t}_{\rm cls}^\top\\
\mathbf{T}_i
\end{bmatrix}
\in\mathbb{R}^{(P+1)\times d_{\rm model}}
```
とする。位置埋め込みは同じ形で

```math
\mathbf{E}_{\rm pos}\in\mathbb{R}^{(P+1)\times d_{\rm model}}
```

を用意し、

```math
\mathbf{U}^{\rm full}_i \leftarrow \mathbf{U}^{\rm full}_i + \mathbf{E}_{\rm pos}
```

CLS+全トークンに位置埋め込みを加算する。

### 3. 可視部分の抽出

可視インデックス集合を $\mathcal{V}\subset\left\{1,\ldots,P\right\}$ を用いて可視トークンを

```math
\mathbf{T}_{\mathcal{V}} = \mathbf{T}_i[\mathcal{V}] \in\mathbb{R}^{|\mathcal{V}|\times d_{\rm model}}
```

と定義する（行の gather）。<br>
このとき、位置埋め込みも同じ行を gather して

```math
\mathbf{E}_{\rm pos,\mathcal{V}} = \mathbf{E}_{\rm pos}\left[\{0\}\cup \mathcal{V}\right]
\in\mathbb{R}^{(|\mathcal{V}|+1)\times d_{\rm model}}
```

$\left\{0\right\}$ は CLS 行のインデックスである。<br>
よって、エンコーダ入力は

```math
\boxed{
\mathbf{U}^{\rm enc}_i
=
\begin{bmatrix}
\mathbf{t}_{\rm cls}^\top\\
\mathbf{T}_{\mathcal{V}}
\end{bmatrix}
+
\mathbf{E}_{\rm pos,\mathcal{V}}
\in\mathbb{R}^{(|\mathcal{V}|+1)\times d_{\rm model}}
}
```

### 4. Encoder

入力 $\mathbf{U}^{\rm enc}_i$ に対し、 $L$ 層の Transformer Encoder を適用して隠れ状態列を得る：

```math
\mathbf{H}_i
=
{\rm Enc}_{\theta}\!\left(\mathbf{U}^{\rm enc}_i\right)
\in\mathbb{R}^{(|\mathcal{V}|+1)\times d_{\rm model}}
```

ここで ${\rm Enc}_{\theta}$ は、同一形状を保つ Self-Attention + FFN ブロックの合成（ $L$ 層）である。<br>
CLS 出力（先頭トークン）を

```math
\mathbf{h}_{\mathrm{cls},i}
=
\mathbf{H}_i^{(L)}[0]
\in\mathbb{R}^{d_{\rm model}}
```

と定義する。

**TransformerEncoderLayer**

各層 $\ell=\left\{1,\ldots,L\right\}$ は

* Multi-Head Self-Attention（MHSA）
* Position-wise FFN
* 残差接続（Residual）
* LayerNorm

から構成され、入力と同じ形状の出力を返す：

```math
\begin{aligned}
\mathbf{H}_i^{(0)}&=\mathbf{U}^{\rm enc}_i,\\
\mathbf{H}_i^{(\ell)}&={\rm TransformerEncoderLayer}^{(\ell)}\!\left(\mathbf{H}_i^{(\ell-1)}\right)
\end{aligned}
```

最終的に $\mathbf{H}_i^{(L)}$ を得る。

**Projection head**

CLS 表現 $\mathbf{h}_{\mathrm{cls},i}$ を潜在表現 $\mathbf{z}_i\in\mathbb{R}^{d_z}$ に線形射影する：

```math
\bar{\mathbf{z}_i}
=
\mathbf{W}_{\rm proj}\mathbf{h}_{\mathrm{cls},i}+\mathbf{b}_{\rm proj}
\in\mathbb{R}^{d_z},
\quad
\mathbf{W}_{\rm proj}\in\mathbb{R}^{d_z\times d_{\rm model}},
\mathbf{b}_{\rm proj}\in\mathbb{R}^{d_z}
```

SNV後のスペクトルと幾何的整合性をとるために $\ell_2$ 正規化し、潜在表現 $\mathbf{z}_i\in\mathbb{S}^{d_z-1}$ を得る：

```math
\mathbf{z}_i
=
\frac{\bar{\mathbf{z}_i}}{\|\bar{\mathbf{z}_i}\|_2}
\in\mathbb{S}^{d_z-1}
```

### 5. Decoder

Encoder で得た潜在表現 $\mathbf{z}_i$ から、元のスペクトル $\mathbf{x}_i\in\mathbb{R}^C$ を再構成するデコーダ ${\rm Dec}_{\phi}$ を定義する。<br>
本研究ではデコーダを MLP とし、 $\mathbf{z}_i$ を直接 $C$ 次元へ写像して再構成スペクトル $\hat{\mathbf{x}}_i$ を得る：

```math
\hat{\mathbf{x}}_i
=
{\rm Dec}_{\phi}(\mathbf{z}_i)
\in\mathbb{R}^{C}
```

### 6. Loss Function

**マスク支持子 $\mathbf{m}$ の作成**

パッチ集合 $\mathcal{M}=\left\{1,\ldots,P\right\}\setminus\mathcal{V}$ からマスク支持子 $\mathbf{m}\in\left\{0,1\right\}^C$ を作成する。<br>
パッチ $j\in\left\{1,2,\ldots,P\right\}$ が覆う波長点インデックス集合を

```math
\mathcal{I}(j)
=
\{(j-1)p+1,(j-1)p+2\ldots, jp\}
```

と定義する。<br>
マスクされた波長点集合は $\bigcup_{j\in\mathcal{M}} \mathcal{I}(j)$ であるから、<br>
波長点レベルのマスク指示子 $\mathbf{m}\in\left\{0,1\right\}^{C}$ を

```math
m_c
=
\begin{cases}
1 & \left(c\in \bigcup_{j\in\mathcal{M}} \mathcal{I}(j)\right)\\
0 & \left(c\notin \bigcup_{j\in\mathcal{M}} \mathcal{I}(j)\right)
\end{cases}
\qquad (c=1,\ldots,C)
```

(※) $\mathbf{m}$ はマスクされたパッチに属する成分だけ 1 になるベクトルである。

**損失関数（マスク領域でのみ計算）**

元スペクトル $\mathbf{x}_i\in\mathbb{R}^C$ 、再構成 $\hat{\mathbf{x}}_i\in\mathbb{R}^C$ に対し、アダマール積 $\odot$ を用いて

```math
\boxed{
\mathcal{L}_{\rm mask}(i)
=
\left\|\mathbf{m}\odot(\hat{\mathbf{x}}_i-\mathbf{x}_i)\right\|_2^2
}
```

---

## 教師なしセグメンテーション

MAE により得られた潜在表現を Cosine K-Means でクラスタリングし、各スペクトル（画素）にクラスタ ID を割り当てる。次に、あらかじめ作成した木材領域の二値マスクを用いて、木材領域にのみクラスタ ID を埋め戻すことでラベル画像を再構成する。具体的には、マスクで抽出した木材画素の1次元配列にクラスタ ID を対応付け、背景画素には無効値 -1 を付与したうえで、元の画像サイズ (H, W) に戻してラベルマップを得る。

<img src="images/clustering.png" width="1000">

### Cosine K-Means

$\mathbf{X}\in\mathbb{R}^{N_{\rm pix}\times C}$ を **全可視**（マスクなし）で学習済み Encoder に入力して得た特徴量行列を $\mathbf{Z}\in\mathbb{R}^{N_{\rm pix}\times d_z}$ とする。<br>
各行ベクトル $\mathbf{z}_i$ は $\ell_2$ 正規化されており $\|\mathbf{z}_i\|_2=1$ を満たすとする。

クラスタ数を $K$ とし、クラスタ中心を行に並べた行列を

```math
\mathbf{C}
=
\begin{bmatrix}
\mathbf{c}_1^\top\\
\vdots\\
\mathbf{c}_K^\top
\end{bmatrix}
\in\mathbb{R}^{K\times d_z},
\qquad
\|\mathbf{c}_k\|_2=1\ (k=1,\ldots,K)
```

で表す。

割当はラベルベクトル $\mathbf{a}=(a_1,\ldots,a_{N_{\rm pix}})^\top\in\left\{1,\ldots,K\right\}^{N_{\rm pix}}$ で表し、 $a_i=k$ が「点 $i$ をクラスタ $k$ に割り当てる」ことを意味する。

このとき目的関数はコサイン距離により

```math
\min_{\mathbf{a},\mathbf{C}}
\frac{1}{N_{\rm pix}}\sum_{i=1}^{N_{\rm pix}}
\left(1-\mathbf{z}_i^\top \mathbf{c}_{a_i}\right)
```

と書ける。同時に最適化するのが難しいため、最適化は交互更新で行う。

まず割当更新（E-step）は、類似度行列

```math
\mathbf{S}=\mathbf{Z}\mathbf{C}^\top\in\mathbb{R}^{N_{\rm pix}\times K}
```

を計算し、

```math
a_i \leftarrow \arg\min_{k\in\{1,\ldots,K\}} \left(1-\mathbf{S}_{ik}\right)
```

で与える。（ $\mathbf{S}_{ik}$ は類似度行列 $\mathbf{S}$ の $(i, k)$ 成分で $\mathbf{S}_{ik}=\mathbf{z}_i^\top\mathbf{c}_k$ ）

次に中心更新（M-step）は、クラスタ $k$ の割当集合（インデックスの部分集合）

```math
\mathcal{S}_k=\{i\in\{1,\ldots,N_{\rm pix}\}\mid a_i=k\}
```

を用いて

```math
\bar{\mathbf{c}}_k
=
\frac{1}{|\mathcal{S}_k|}\sum_{i\in\mathcal{S}_k}\mathbf{z}_i,
\qquad
\mathbf{c}_k \leftarrow \frac{\bar{\mathbf{c}}_k}{\|\bar{\mathbf{c}}_k\|_2}
```

と更新する。

以上により、埋め込み $\mathbf{Z}$ の球面幾何に整合したクラスタリングが得られる。

---

## 付録

**試験条件**

| Temperature | t1  | t2    | t3    | t4    | t5    | t6    | t7    | t8    | t9    |
|-------------|-----|-------|-------|-------|-------|-------|-------|-------|-------|
| $120^{\circ}\mathrm{C}$ | 0h  | 4d    | 8d    | 16d   | 32d   | 64d   | 128d  | 256d  | 512d  |
| $140^{\circ}\mathrm{C}$ | 0h  | 12h   | 1d    | 2d    | 4d    | 8d    | 16d   | 32d   | 64d   |
| $160^{\circ}\mathrm{C}$ | 0h  | 3h    | 6h    | 12h   | 1d    | 2d    | 4d    | 8d    | 16d   |
| $180^{\circ}\mathrm{C}$ | 0h  | 0.75h | 1.5h  | 3h    | 6h    | 12h   | 1d    | 2d    | 4d    |

(※) h: hour, d: day

**クスノキ**

| A1 | A2 | A3 |
|----|----|----|
| <img src="./images/A1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/A2_cluster_labels_latent_ckm.png" width="300"> | <img src="./images/A3_cluster_labels_latent_ckm.png" width="300"> |

**クリ**

| B1 | B2 | B3 |
|----|----|----|
| <img src="./images/B1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/B2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/B3_cluster_labels_latent_ckm.png"  width="300"> |

**ヒノキ**

| C1 | C2 | C3 |
|----|----|----|
| <img src="./images/C1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/C2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/C3_cluster_labels_latent_ckm.png"  width="300"> |

**マツ**

| D1 | D2 | D3 |
|----|----|----|
| <img src="./images/D1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/D2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/D3_cluster_labels_latent_ckm.png"  width="300"> |

**ヤマザクラ**

| E1 | E2 | E3 |
|----|----|----|
| <img src="./images/E1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/E2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/E3_cluster_labels_latent_ckm.png"  width="300"> |

**ライムウッド**

| V1 | V2 | V3 |
|----|----|----|
| <img src="./images/V1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/V2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/V3_cluster_labels_latent_ckm.png"  width="300"> |

**ハードメープル**

| W1 | W2 | W3 |
|----|----|----|
| <img src="./images/W1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/W2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/W3_cluster_labels_latent_ckm.png"  width="300"> |

**ポプラ**

| X1 | X2 | X3 |
|----|----|----|
| <img src="./images/X1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/X2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/X3_cluster_labels_latent_ckm.png"  width="300"> |

**スプルース**

| Y1 | Y2 | Y3 |
|----|----|----|
| <img src="./images/Y1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/Y2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/Y3_cluster_labels_latent_ckm.png"  width="300"> |

**オーク**

| Z1 | Z2 | Z3 |
|----|----|----|
| <img src="./images/Z1_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/Z2_cluster_labels_latent_ckm.png"  width="300"> | <img src="./images/Z3_cluster_labels_latent_ckm.png"  width="300"> |

**クラスタスペクトル**

- 左上：Reflectance, 右上：Reflectance(SNV), 左下：Absorbance, 右下: Absorbance(SNV)

<img src="./images/cluster_spectra.png" width="1000">

- Absorbanceの二次微分（SG）

<img src="./images/cluster_spectra_2nd_derive.png" width="1000">