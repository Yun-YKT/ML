# LSTM
RNNよりも長期的な関係を学習可能にしたモデル．  
内部にゲート構造を持つことにより，これを可能にしている．  
詳しくは[LSTMネットワークの概要（Qiita記事）](https://qiita.com/KojiOhki/items/89cd7b69a8a6239d67ca)を参照

## コードについて
LSTMをpytorchで実装するにあたってミニマムな実装をわかりやすく解説してくれている以下のサイトを参考にした．  
[pytorch で LSTM に入門したい...したくない？](https://hilinker.hatenablog.com/entry/2018/06/23/204910)  

### パラメータ
-ms int型, default=1 (modelへのinputサイズ)
-bs int型, default=config.pyのBATCH_SIZE　（データのbatchサイズ）
-hs int型, default=100　（iddenlayerのサイズ)
-ne int型, default=config.pyのEPOCHS　（学習のepoch数）
~~-ng int型, default=4　（学習に使うGPUの数）~~ 現在未実装
-lr float型, default=1e-4　（SGDの学習率）
~~-data_dir　~~ 現在未実装

### 入力データ
LSTM内のbatch_firstパラメータをTrueにしているため，入力のtensorは **\[バッチサイズ * 時系列長 * 入力テンソル]** になることに注意．

## 実行方法
正弦波＋ノイズでの学習をしたい場合は，
~~~ 
python train.py
~~~
~~学習データを任意のデータで行いたい場合は，~~ 現在未実装
~~~
python train.py -data_dir ~/ファイルへのパス
~~~
その他のパラメータ（epoch数やbatchサイズ）を変更したい場合は，
~~~
python train.py -bs 任意のbatchサイズ -ne 任意のepoch数
~~~

