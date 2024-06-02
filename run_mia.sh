# for i in {0..12}
# do
# for g in {0..9}
# do
# CUDA_VISIBLE_DEVICES=$g python examples/bert_cls.py --seed ${i}${g} --dp False  --dataset_name sst2 --batch_size 8 --model_name  bert-large-uncased &
# done
# wait
# done

# wait


# for i in {0..13}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning.py --seed ${i}${g}   --dataset_name sst2 --batch_size 8 --tuning_method prefix-tuning --model_name bert-large-uncased &
#  done
# wait
# done

# for i in {0..13}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning.py --seed ${i}${g}   --dataset_name qnli --batch_size 8 --tuning_method prefix-tuning --model_name bert-base-uncased &
#  done
# wait
# done

# full finetune
# for i in {0..6}
# do
# for g in {0..9}
# do
# CUDA_VISIBLE_DEVICES=$g python examples/bert_cls.py --seed ${i}${g} --dp True  --dataset_name sst2 --batch_size 8  --model_name bert-base-uncased &
# done
# wait
# done

# for i in {0..12}
# do
# for g in {0..9}
# do
# CUDA_VISIBLE_DEVICES=$g python examples/bert_cls.py --seed ${i}${g} --dp True  --dataset_name sst2 --batch_size 8  --model_name roberta-base &
# done
# wait
# done



for i in {0..9}
do
for g in {0..9}
do
 CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method p-tuning --model_name bert-large-uncased --dp True &
 done
wait
done

for i in {0..9}
do
for g in {0..9}
do
 CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 8 --tuning_method p-tuning --model_name bert-base-uncased --dp True &
 done
wait
done

# # roberta large non-dp
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method p-tuning --model_name roberta-large  &
#  done
# wait
# done

# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 8 --tuning_method p-tuning --model_name roberta-base  &
#  done
# wait
# done


# for i in {0..9}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 1 --model_name bert-base-uncased --dp False &
#  done
# wait
# done


# for i in {0..9}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 1 --model_name bert-large-uncased --dp False &
#  done
# wait
# done


# wait



# # roberta large non-dp p-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method p-tuning --model_name roberta-large  &
#  done
# wait
# done

# # roberta large dp prefix-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method prefix-tuning --model_name roberta-large --dp True &
#  done
# wait
# done

# # roberta base non-dp p-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method p-tuning --model_name roberta-base  &
#  done
# wait
# done

# # roberta base dp p-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method p-tuning --model_name roberta-base --dp True &
#  done
# wait
# done

# # roberta base dp prefix-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method prefix-tuning --model_name roberta-base --dp True &
#  done
# wait
# done

# # bert large dp prefix-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method prefix-tuning --model_name bert-large-uncased --dp True &
#  done
# wait
# done

# # bert base nondp p-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method prefix-tuning --model_name bert-base-uncased &
#  done
# wait
# done

# # roberta base dp p-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method p-tuning --model_name roberta-base --dp True &
#  done
# wait
# done

# # bert large prefix dp 
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning_mia.py --seed ${i}${g}   --dataset_name qnli --batch_size 4 --tuning_method prefix-tuning --model_name bert-large-uncased --dp True &
#  done
# wait
# done



# # bert base nondp p-tuning
# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py  --seed ${i}${g}   --dataset_name qnli --batch_size 4  --model_name bert-base-uncased &
#  done
# wait
# done

# # for i in {0..4}
# # do
# # for g in {0..9}
# # do
# #  echo ${i}${g}   
# #  done
# # wait
# # done

# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py  --seed ${i}${g}   --dataset_name qnli --batch_size 4  --model_name bert-large-uncased &
#  done
# wait
# done


# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py  --seed ${i}${g}   --dataset_name qnli --batch_size 4  --model_name roberta-base --dp True&
#  done
# wait
# done


# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py  --seed ${i}${g}   --dataset_name qnli --batch_size 4  --model_name roberta-large --dp True &
#  done
# wait
# done

# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py  --seed ${i}${g}   --dataset_name qnli --batch_size 4  --model_name bert-large-uncased --dp True &
#  done
# wait
# done


# for i in {0..4}
# do
# for g in {0..9}
# do
#  CUDA_VISIBLE_DEVICES=$g python examples/bert_mlm_mia.py  --seed ${i}${g}   --dataset_name qnli --batch_size 4  --model_name roberta-base --dp True &
#  done
# wait
# done
