#  KGA: A General Machine Unlearning Framework Based on Knowledge Gap Alignment (ACL 2023)
[[Paper]](https://aclanthology.org/2023.acl-long.740.pdf) 

## Requirement:

* Python: 3.6+
* Pytorch: 1.8.1+

## Data Preparation:

Classification: LEDGAR (Please refer to the [Format](https://github.com/dtuggener/LEDGAR_provision_classification/))

Translation: IWSLT De-En (Format: xxx.de and xxx.en two files, parallel lines)

Response generation: PersonaChat (Format: xxx.src and xxx.tgt, same as translation)

## Usage Examples:

### Classification:

Train:

```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --data [data-path] \
    --do_train \
    --mode train-forget [or train-new] \
    --file_as_new [Indices file for new] \
    --file_removals [Indices file for forget] \
    --save_path [model save file name] \
    --batch_size 32 \
    --sample_ratio 0.05 \
    --learning_rate 2e-5 \
    --warmup_steps 500 
```

Unlearn:

```
CUDA_VISIBLE_DEVICES=0 python kga_classification.py \
    --data [data path] \
    --do_unlearn \
    --file_as_new [Indices file for new] \
    --file_removals [Indices file for forget] \ 
    --save_path [model save path] \
    --model_path [original model path] \
    --new_model_path [new model path] \
    --forget_model_path [forget model path] \
    --batch_size 32 \
    --sample_ratio 0.3 \ 
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --retain_loss_ratio 0.1 \ 
    --inner_step 10 \ 
    --print_loss 200 \ 
    --eval_update 1000 \ 
    --save_update 5000
```

### Translation:

Train:

```
CUDA_VISIBLE_DEVICES=0 python run_translation.py \
    --output_dir [model save path] \
    --model_type marian \
    --model_checkpoint opus_mt_de_en \
    --train_file [train data path] \
    --dev_file [dev data path] \
    --do_train \
    --batch_size 32 \
    --update_freq 16 \
    --learning_rate 5e-4 \
    --num_train_epochs 50 \
    --warmup_steps 4000 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --lr_schedule "inverse_sqrt" \
    --beam 5 \
    --source de --target en
```

Unlearn:

```
CUDA_VISIBLE_DEVICES=0 python kga_translation.py \
    --output_dir [model save path] \
    --new_model_dir [new model path] \
    --forget_model_dir [forget model path] \
    --train_model_dir [original model path] \
    --model_type marian \
    --model_checkpoint opus_mt_de_en \
    --train_file [train data path] \
    --dev_file [dev data path] \
    --forget_file [forget data path] \
    --new_file [new data path] \
    --do_unlearn \
    --retain_loss_ratio 0.3 \
    --batch_size 16 \
    --update_freq 8 \
    --learning_rate 2e-5 \
    --num_train_updates 5000 \
    --warmup_steps 1000 \
    --weight_decay 0.0001 \
    --lr_schedule "inverse_sqrt" \
    --beam 5 \
    --source de --target en
```

### Response Generation:

Train:

```
CUDA_VISIBLE_DEVICES=0 python run_generation.py \
    --output_dir [model save path] \
    --model_checkpoint bart-base \
    --train_file [train data path] \
    --dev_file [dev data path] \
    --do_train \
    --batch_size 16 \
    --update_freq 2 \
    --learning_rate 2e-4 \
    --num_train_epochs 100 \
    --warmup_steps 4000 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --lr_schedule "inverse_sqrt" \
    --beam 5
```

Unlearn:

```
CUDA_VISIBLE_DEVICES=0 python kga_generation.py \
    --output_dir [model save path] \
    --new_model_dir [new model path] \
    --forget_model_dir [forget model path] \
    --train_model_dir [original model path] \
    --model_checkpoint bart-base \
    --train_file [train data path] \
    --dev_file [dev data path] \
    --forget_file [forget data path] \
    --new_file [new data path] \
    --do_unlearn \
    --retain_loss_ratio 0.3 \
    --batch_size 16 \
    --update_freq 2 \
    --learning_rate 5e-5 \
    --num_train_updates 12000 \
    --num_train_epochs 50 \
    --stop_value 0.05 \
    --warmup_steps 1000 \
    --weight_decay 0.0001 \
    --lr_schedule "inverse_sqrt" \
    --beam 5
```

## Reference
If you find our paper helpful and use this code, please cite our publication at ACL 2023. 

```
@inproceedings{wang-etal-2023-kga,
    title = "{KGA}: A General Machine Unlearning Framework Based on Knowledge Gap Alignment",
    author = "Wang, Lingzhi  and
      Chen, Tong  and
      Yuan, Wei  and
      Zeng, Xingshan  and
      Wong, Kam-Fai  and
      Yin, Hongzhi",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.740",
    doi = "10.18653/v1/2023.acl-long.740",
    pages = "13264--13276",
}
```
