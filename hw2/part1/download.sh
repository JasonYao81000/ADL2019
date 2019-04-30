#!/usr/bin/env bash
curl -o ./data/classification/config.yaml https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/config.yaml
curl -o ./data/classification/char.pkl https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/char.pkl
curl -o ./data/classification/word.pkl https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/word.pkl
curl -o ./data/classification/test.csv https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/test.csv
curl -o ./model/submission/ckpts/epoch-9.ckpt https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/epoch-9.ckpt
curl -o ./model/submission/config.yaml https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/model/config.yaml
curl -o ./ELMo/configs/cnn_50_100_512_4096_sample.json https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/cnn_50_100_512_4096_sample.json
curl -o ./ELMo/output/char.dic https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/char.dic
curl -o ./ELMo/output/word.dic https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/word.dic
curl -o ./ELMo/output/config.json https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/config.json
curl -o ./ELMo/output/encoder.pkl https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/encoder.pkl
curl -o ./ELMo/output/token_embedder.pkl https://raw.githubusercontent.com/wangssuming/adlhw2_model/master/part1/token_embedder.pkl