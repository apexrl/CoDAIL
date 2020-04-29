#! /bin/bash
# source activate rl
python -m irl.render --env=simple_speaker_listener --algo=mack_om --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_speaker_listener --algo=codail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_speaker_listener --algo=ncdail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_speaker_listener --algo=gail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_speaker_listener --algo=airl --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_speaker_listener --algo=random --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3

python -m irl.render --env=simple_spread --algo=mack_om --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_spread --algo=codail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_spread --algo=ncdail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_spread --algo=gail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_spread --algo=airl --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_spread --algo=random --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3

python -m irl.render --env=simple_push --algo=mack_om --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_push --algo=codail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_push --algo=ncdail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_push --algo=gail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_push --algo=airl --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_push --algo=random --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3

python -m irl.render --env=simple_tag --algo=mack_om --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_tag --algo=codail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_tag --algo=ncdail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_tag --algo=gail --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_tag --algo=airl --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3
python -m irl.render --env=simple_tag --algo=random --num_trajs=$1 --epoch=$2 --vis_dis --seed=$3

