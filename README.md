# BEAST
An online beat tracking system based on streaming Transformer.

[![arXiv](https://img.shields.io/badge/arXiv-2312.17156-b31b1b.svg)](https://arxiv.org/abs/2312.17156)

The source code of the paper [BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer](https://arxiv.org/abs/2312.17156), accepted by ICASSP 2024.

More information to be updated.

## Usage
1. Data preparation

If you want to directly evaluate the model, you can download the preprocessed GTZAN data [here](https://drive.google.com/file/d/1BQDXxCYxhU6iFC5wb_W6QYXjWUE5PUXO/view?usp=sharing) and the pretrained model [here](https://drive.google.com/file/d/17yiv4cIsI1rBL8vUAtAVJN1pUPXQOAhl/view?usp=sharing). And put the model under the `./data` directory.

2. Evaluation

Run `./code/eval.py`.

## Contact
- Chih-Cheng Chang (ccchang12@iis.sinica.edu.tw)

## Acknowlegement
1. We borrowed the code from [ESPnet](https://github.com/espnet/espnet) for contextual block processing transformer based modeling.
2. We borrowed the code from [Beat-Transformer](https://github.com/zhaojw1998/Beat-Transformer) for beat tracking.
