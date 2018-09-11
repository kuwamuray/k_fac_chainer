K-FAC実装への道のり
0. フレームワーク選定（or 自前）
1. まとめを読む
2. 自然勾配法
   2-1. exact Fisherを使う
   2-2. empirical Fisherを使う
3. fc層のためのkfac実装
   3-0. forward/backward計算で生成されるa,gの取得
   3-1. a,g の直積の期待値A,Gの計算
   3-2. A,Gの逆行列A_inv, G_invの計算
   3-3. A_inv, G_inv, dL/dθ を用いた自然勾配（近似）= kfgradの計算
   3-4. θの更新
4. conv層のためのkfac実装
   4-1. まとめを読んで近似・計算を理解
   4-2. fc層と同様
5. damping実装
   5-0. 論文中のdampingの記述を探す
   5-1. πの計算法の理解
   5-2. πで調整されたdampingを算出
   5-3. dampingをA,Gに足しこんでinvを計算
6. L2正則化と組み合わせる
   6-1. damping =（L2正則化の係数 + damping）で5-1 - 5-3 を行う (edited)

oosawak [12:10 PM]
実装が進み次第適宜以下を実験
1. A,GがPSDとなっていることを確かめる
   1-0. A,Gが対称行列であることを確認する
   1-1. A,Gの固有値を調べる
   1-2. A,GがPSDになっていないなら、dampingを調整してみる
   1-3. A_inv, G_invの固有値を調べる（全部正であってほしい）
2. 学習をSGD, Adamと比較
   2-1. MNIST（MLP）
   2-2. CIFAR10（AlexNet）
   2-3. CIFAR10（ResNet-50, BN層はSGDで学習 or 学習しない）
   2-4. 上記学習のハイパーパラメータ調整をして学習結果を観察する（各hpの効果の感覚を得る）
3. 学習中の値の統計（mean, std）を観察
   3-0. ベイズ統計を勉強（教材は保留）
   3-1. θのノルムを観察
   3-2. kfgradのノルムを観察
   3-3. （θのノルム）/（kfgradのノルム）を観察
   3-4. 学習率、L2正則化、dampingとこれらの値との関係を観察する (edited)

oosawak [12:24 PM]
K-FAC実装が終わったら、やっていきたいこと
1. K-FAC for RNNの実装
2. K-FAC for 強化学習の実装
3. K-FAC for VI（変分推論）の実装
4. 上記の分散K-FACとの組み合わせ
5. Fisherの非対角ブロック行列の値の観察
6. 平均場近似理論の話が実用的なネットワーク（ResNetなど）へ適用可能かを研究
7. F_invのブロック三重対角近似の実装
8. K-FACと正規化手法（BatchNormなど）との関係を観察
9. パラメータ数の少ないネットワーク（ResNextなど）へのK-FACの適用（inv計算がかなり楽になる＆収束は速い）
@kuwamuray  Chainer
@Hikaru PyTorch
