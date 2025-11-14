

This repository is the PyTorch implementation for AAAI 2025 Paper 
"A Comprehensive and Simple Benchmark for Single-Cell Transcriptomics",

BIBM 2025 Paper "Graph Neural Networks as a Substitute for Transformers in Single-Cell Transcriptomics"

and AAAI 2026 Paper "Gene Incremental Learning for Single-Cell Transcriptomics"

For detailed information, please refer to our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/32049).

Note that this repository is based on the [scGPT](https://github.com/bowang-lab/scGPT), especially the data downloading and preprocessing part.

If you find this work useful in your research, please kindly consider citing:
```
@inproceedings{qi2025scbenchmark,
  title={A Comprehensive and Simple Benchmark for Single-Cell Transcriptomics},
  author={Qi, Jiaxin and Cui, Yan and Guo, Kailei and Zhang, Xiaomin and Huang, Jianqiang and Xie, Gaogang},
  booktitle={AAAI},
  year={2025}
}
@inproceedings{qi2025scgnn,
  title={Graph Neural Networks as a Substitute for Transformers in Single-Cell Transcriptomics},
  author={Qi, Jiaxin and Cui, Yan and Ou, Jinli and Huang, Jianqiang},
  booktitle={BIBM},
  year={2025}
}
@inproceedings{qi2025scincre,
  title={Gene Incremental Learning for Single-Cell Transcriptomics},
  author={Qi, Jiaxin and Cui, Yan and Huang, Jianqiang and Xie, Gaogang},
  booktitle={AAAI},
  year={2026}
}
```
## Update (2025.11)
For BIBM 2025, please just change --model_structure transformer as gnn_cp (gnn_for_compare), whose performance is comparable to Transformers!

For AAAI 2026, the incremental setting, first download a class-wise gene importance file [gene_im](https://drive.google.com/file/d/1u5A7utwdnm9qXW7X1jFkHB5f8L6EP9-V/view?usp=drive_link), and add to the dict ./analysis/ (used in utils.py)
Then, please run the following for incremental learning model training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --nnodes=1   --node_rank=0  --master_addr="localhost"   --master_port=6200 incremental_train.py -s ./save_incre/temp --use_weighted_sampling  --num_stages 3  --select_idxnum 150 --keep_base_id  --filter_names norman lupus --incre_only --epochs 5 --lr 5e-4 --excludefin
```
Then, run the following for evaluations (for the same incremental datasets)
```
CUDA_VISIBLE_DEVICES=7 python incre_downstreams.py --model_dir ./save_incre/temp   --train_from_features --use_weighted_sampling --num_trials 3 --filter_names norman lupus
```
### Prepare Data and Env

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

When prepare environment, please install the suitable version of pytorch (the default is 2.3.1) and refer to requirements.txt to install others, and note that datasets=2.16.0 is essential.

For example, for CUDA 12.4:
```
pip install torch==2.3.1+cu124 torchvision==0.15.1+cu124 torchaudio==2.3.1 -f https://download.pytorch.org/whl/cu124/torch_stable.html
pip install -r requirements.txt
```
for further prepare your data, please refer to scGPT to set your environment.
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
