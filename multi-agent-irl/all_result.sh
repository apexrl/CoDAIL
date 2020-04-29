#! /bin/bash
# source activate rl
if [ $1 -eq 200 ]
then
    python -m irl.render --env=simple_speaker_listener --all_exp --epoch=$2 --traj_limitation=200
    python -m irl.render --env=simple_spread --all_exp --epoch=$2 --traj_limitation=200
    python -m irl.render --env=simple_push --all_exp --epoch=$2 --traj_limitation=200
    python -m irl.render --env=simple_tag --all_exp --epoch=$2 --traj_limitation=200
elif [ $1 -eq 100 ]
then
    python -m irl.render --env=simple_speaker_listener --all_exp --epoch=$2 --traj_limitation=100
    python -m irl.render --env=simple_spread --all_exp --epoch=$2 --traj_limitation=100
    python -m irl.render --env=simple_push --all_exp --epoch=$2 --traj_limitation=100
    python -m irl.render --env=simple_tag --all_exp --epoch=$2 --traj_limitation=100
fi
