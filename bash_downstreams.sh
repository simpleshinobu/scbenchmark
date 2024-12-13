#!/bin/bash
declare -a model_paths=(
    "baseline/best_model.pt"
)
declare -a nlayers=(6)
declare -a embsize=(256)
export CUDA_VISIBLE_DEVICES=0
filter_names=("Myeloid" "Multiple_Sclerosis" "pancread" "ircolitis" "myasthenia" "lupus" "scfoundation" "scanorama" "dengue" "leptomeningeal")

for index in "${!model_paths[@]}"; do
    model_path=${model_paths[$index]}
    nlayer=${nlayers[$index]}
    embs=${embsize[$index]}
    for filter_name in "${filter_names[@]}"; do
        echo "Running model $model_path with filter $filter_name"
        python downstreams_cls.py --model_path ./save_pretrain/${model_path} \
                                  --nlayers $nlayer --embsize $embs \
                                  --filter_name $filter_name --eval_knn --train_from_features \
                                  --cell_emb_style cls --use_weighted_sampling --num_trials 5 --test_maxseq 512 \
                                  --model_structure transformer
    done
done
filter_names=( "adamson" "dixit" "norman")
for index in "${!model_paths[@]}"; do
    model_path=${model_paths[$index]}
    nlayer=${nlayers[$index]}
    embs=${embsize[$index]}
    for filter_name in "${filter_names[@]}"; do
        echo "Running model $model_path with filter $filter_name"
        python downstreams_cls2.py --model_path ./save_pretrain/${model_path} \
                                  --nlayers $nlayer --embsize $embs \
                                  --filter_name $filter_name --train_from_features \
                                  --cell_emb_style cls --use_weighted_sampling --num_trials 5 --test_maxseq 512 \
                                  --model_structure transformer
    done
done
for index in "${!model_paths[@]}"; do
    model_path=${model_paths[$index]}
    nlayer=${nlayers[$index]}
    embs=${embsize[$index]}
    for filter_name in "${filter_names[@]}"; do
        echo "Running model $model_path with filter $filter_name"
        python downstreams_perturbe_pred.py --model_path ./save_pretrain/${model_path} \
                                  --nlayers $nlayer --embsize $embs \
                                  --filter_name $filter_name  --use_weighted_sampling --print_epoch 10 \
                                  --model_structure transformer --num_trials 5
    done
done
