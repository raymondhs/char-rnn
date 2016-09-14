set -e -x

# wiki, wsj, conll_eng, conll_deu
prefix=$1 

# lstm, gru
model=$2

# small, large
size=$3

# >= 0 for GPU, -1 for CPU
gpu=$4

if [ "$#" -ne 4 ] ; then
  echo "Usage: $0 <corpus> [lstm|gru] [small|large] <gpuid>" >&2
  exit 1
fi

# Hyperparameters
drop=0.25
batch_size=100
seq_len=50
data=data/$prefix
max_epochs=30
learning_rate=0.002
rnn=300
layer=2
if [ $size = "large" ]; then
  rnn=700
  layer=3
  if [ $model = "gru" ]; then
    learning_rate=0.001
  fi
fi

cv_dir=cv/"$prefix"_"$model"_"$rnn"hidden_"$layer"layer
mkdir -p cv/"$prefix"_"$model"_"$rnn"hidden_"$layer"layer
rm -f $cv_dir/*

echo $model $size
time th train.lua \
-data_dir $data \
-model $model \
-rnn_size $rnn \
-num_layers $layer \
-dropout $drop \
-batch_size $batch_size \
-seq_length $seq_len \
-max_epochs $max_epochs \
-learning_rate $learning_rate \
-checkpoint_dir $cv_dir \
-gpuid $gpu > $cv_dir/train.out 2> $cv_dir/train.err

