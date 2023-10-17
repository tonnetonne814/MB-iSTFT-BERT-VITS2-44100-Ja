# MB-iSTFT-BERT-VITS2
[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)のデコーダをMB-iSTFT-VITSのデコーダに変更したもの。**実験用です。**自分の環境で動くことを最優先にして編集してあります。

## 1. 環境構築

Anacondaによる実行環境構築を想定する。

0. Anacondaで"MB-iSTFT-BERT-VITS2"という名前の仮想環境を作成する。[y]or nを聞かれたら[y]を入力する。
    ```sh
    conda create -n MB-iSTFT-BERT-VITS2 python=3.8    
    ```
0. 仮想環境を有効化する。
    ```sh
    conda activate MB-iSTFT-BERT-VITS2 
    ```
0. このレポジトリをクローンする（もしくはDownload Zipでダウンロードする）

    ```sh
    git clone https://github.com/tonnetonne814/MB-iSTFT-BERT-VITS2-44100-Ja.git
    cd MB-iSTFT-BERT-VITS2-44100-Ja # フォルダへ移動
    ```

0. [https://pytorch.org/](https://pytorch.org/)のURLよりPyTorchをインストールする。
    
    ```sh
    # OS=Linux, CUDA=11.7 の例
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

0. その他、必要なパッケージをインストールする。
    ```sh
    pip install -r requirements.txt 
    ```

0. [HuggingFace](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)よりBERTのモデルデータをダウンロードし、以下に配置する。
    ```sh
    ./bert/bert-base-japanese-v3/pytorch_model.bin
    ```


## 2. データセットの準備
書き起こしテキストファイルとwavファイルが入ったフォルダを用意する。書き起こしテキストは「wavファイル名|書き起こし文」と記述したもの。dataset_nameの名前は話者名にも流用するようになっています。追記式になってるので、色々なデータセットを一つ一つ追加する。書き起こしテキストのパスは、テキストファイルが入っているフォルダでも可。

```sh
python3 preprocess.py --dataset_name name --dataset_folder path/to/wav/folder --dataset_language JP --text_path path/to/text.txt --split_symbol | 
```
必要なデータセットを追加し終えたら、以下を実行する。
```sh
python3 preprocess_text.py
```

## 4. 学習
次のコマンドを入力することで、学習を開始する。
> ⚠CUDA Out of Memoryのエラーが出た場合には、config.jsonにてbatch_sizeを小さくする。

    ```sh
    python train_ms.py -c configs/jsut_44100.json -m ExpName
    ```

## 5.推論
webuiで動かす予定。そのままで動くはず...