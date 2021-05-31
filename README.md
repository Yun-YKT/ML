# ML

- 就活,インターンに際してGitHubの提出を求められることが多いので，復習も兼ねて機械学習の基礎的内容を1つずつまとめていく．
- コード全体の雛形は[WaveGANをpytorchで実装したもの[github]](https://github.com/mazzzystar/WaveGAN-pytorch)の書き方を参照

## ファイル構成
── Model名（LSTMなど）  
    ├── tarin.py(学習の流れ全般のコード．どのコードもこのファイルをコンパイルすることで実行できる)  
    ├── Model名.py(モデルの構造に関するコード)  
    ├── utils.py(データ処理周りに関するコード)  
    ├── config.py(バッチサイズや，モデルサイズ等のよく変える設定を記述したコード)  
    ├── logger.py（ログに関するコード）  
    └── README.md  
    
