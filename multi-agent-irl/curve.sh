#!/bin/bash
# source activate rl
if [ $1 -eq 200 ]
then
    python -m irl.render --env=simple_speaker_listener --curve --epoch=$2 --traj_limitation=200
    python -m irl.render --env=simple_spread --curve --epoch=$2 --traj_limitation=200
    python -m irl.render --env=simple_push --curve --epoch=$2 --traj_limitation=200
    python -m irl.render --env=simple_tag --curve --epoch=$2 --traj_limitation=200
elif [ $1 -eq 100 ]
then
    python -m irl.render --env=simple_speaker_listener --curve --epoch=$2 --traj_limitation=100
    python -m irl.render --env=simple_spread --curve --epoch=$2 --traj_limitation=100
    python -m irl.render --env=simple_push --curve --epoch=$2 --traj_limitation=100
    python -m irl.render --env=simple_tag --curve --epoch=$2 --traj_limitation=100
fi
