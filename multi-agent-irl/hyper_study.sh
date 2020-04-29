#! /bin/bash
# source activate rl
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=1 --g=2 --ent_coef=0 
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=1 --g=4 --ent_coef=0 
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=2 --g=1 --ent_coef=0 
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=4 --g=1 --ent_coef=0 

python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=1 --g=1 --ent_coef=0.2
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=1 --g=1 --ent_coef=0.4
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=1 --g=1 --ent_coef=0.6
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=1 --g=1 --ent_coef=0.8
python -m irl.render --env=simple_spread --algo=codail --num_trajs=200 --hyper_study --epoch=$1 --d=1 --g=1 --ent_coef=1.0


