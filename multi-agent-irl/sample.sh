#! /bin/bash
# source activate rl
python -m irl.render --env=simple_speaker_listener --algo=mack_om --epoch=$2 --num_trajs=$1 --sample --seed=$3
python -m irl.render --env=simple_spread --algo=mack_om --epoch=$2 --num_trajs=$1 --sample --seed=$3
python -m irl.render --env=simple_push --algo=mack_om --epoch=$2 --num_trajs=$1 --sample --seed=$3
python -m irl.render --env=simple_tag --algo=mack_om --epoch=$2 --num_trajs=$1 --sample --seed=$3
