

This repository is the PyTorch implementation for AAAI 2025 Paper 
"A Comprehensive and Simple Benchmark for Single-Cell Transcriptomics",
For the detailed information, please refer to our paper. 
If you have any questions or suggestions, 
please email me.

Note that this repository is based on the [scGPT](https://github.com/bowang-lab/scGPT), especially the data downloading and preprocessing part.

If you find this work useful in your research, please kindly consider citing:
```
@inproceedings{qi2025scbenchmark,
  title={A Comprehensive and Simple Benchmark for Single-Cell Transcriptomics},
  author={Qi, Jiaxin and Cui, Yan and Guo, Kailei and Zhang, Xiaomin and Huang, Jianqiang and Xie, Gaogang},
  booktitle={AAAI},
  year={2025}
}
```
### Prepare Data

Directly download our preprocessed data:
- [train_data](https://drive.google.com/file/d/1u2I18NfBUTBZY_gUmlXWcoibaLyPy1WT/view?usp=sharing) and [test_data](https://drive.google.com/file/d/1yMX5gMmj3npBUN8lzQx7gHe6MBlTj9oX/view?usp=sharing) into the `./data/` directory.
- [Downstream cls](https://drive.google.com/file/d/1JyUrqOFs1ZskrEHvWR0SO378J0KhEkVG/view?usp=sharing) and [Downstream pert](https://drive.google.com/file/d/1M0CNeJ9_K1x_BwJX0Iwy3-0G5isICffI/view?usp=sharing) into the `./data/downstreams/classification` and `./data/downstreams/perturbation` directory. (e.g., data/downstreams/classification/processed_data/ircolitis_data.pt)

Alternatively, create the pretraining dataset from scratch (several hours).
Run the following command:
```
cd data/cellxgene
python download_data.py --index-dir ./idx_dir --output-dir ./data_raw --vocab-file vocab.json --scoutput-path ./data_scbank --link-dir ./all_counts
```
Then, create your own train data and test data. Note that, downloading the whole data could easily test the Table 6 in the main paper.

### Pretrain Models

Run the baseline model (last lline in Table 1 in the main paper) (6 layers, 256 dimensions, use mvc, use weighted sampling, cls as cell feature, log1p preprocess) with the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=6667 train_baseline.py --epochs 10 --batch_size 128 --lr 2e-4 -s ./save_pretrain/baseline --nlayers 6 --embsize 256 --preprocess_mode none --use_weighted_sampling --use_mvc --model_structure transformer
```
Note that the evaluation regression loss is only for reference and not reported in the main paper. Please change --model_structure to test other structures (for cnn and linear, please add --train_maxseq 511 --test_maxseq 511 for valid input), change --preprocess_mode --use_weighted_sampling --use_mvc to test other settings.

### Downstream Tasks

Run the following command for downstream tasks (change model path before running):
```
bash bash_downstreams.sh
```
Please refer to the "Expression Classification Datasets" section in the main paper to ensure that the dataset names in the code correspond with those in the paper.
### Acknowledgements

Thanks for the source code from scGPT.