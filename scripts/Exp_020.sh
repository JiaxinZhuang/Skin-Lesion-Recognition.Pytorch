set -e
set -x
export PYTHONPATH='./src'

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}


CUDA_VISIBLE_DEVICES=7 python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cudas=0 \
    --n_epochs=2000 \
    --batch_size=96 \
    --server=lab_center \
    --eval_frequency=10 \
    --backbone=resnet50 \
    --learning_rate=1e-4 \
    --optimizer=SGD \
    --initialization=pretrained \
    --num_classes=7 \
    --num_worker=6 \
    --input_channel=3 \
    --iter_fold=1 \
    --seed=1024 \
    2>&1 | tee $log_file
