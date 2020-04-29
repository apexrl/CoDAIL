#! /bin/bash
# source activate rl
python -m irl.render --env=simple_speaker_listener --kl --num_trajs=$1 --epoch=$2 --seed=$3
python -m irl.render --env=simple_spread --kl --num_trajs=$1 --epoch=$2 --seed=$3
python -m irl.render --env=simple_push --kl --num_trajs=$1 --epoch=$2 --seed=$3
python -m irl.render --env=simple_tag --kl --num_trajs=$1 --epoch=$2 --seed=$3

