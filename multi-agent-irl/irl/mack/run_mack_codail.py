#!/usr/bin/env python
import logging
import os
import itertools
import click
import gym

import make_env
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from irl.dataset import MADataSet
from irl.mack.codail import learn
from sandbox.mack.policies_om import CategoricalPolicy as CategoricalPolicy_om
from sandbox.mack.opponent_policies import CategoricalPolicy as oppo_CategoricalPolicy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500, d_iters=1, g_iters=1, ent_coef=0.0):
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    print(num_cpu)
    policy_fn = CategoricalPolicy_om
    oppo_policy_fn = oppo_CategoricalPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation)
    learn(policy_fn, oppo_policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.0, dis_lr=dis_lr,
          disc_type=disc_type, bc_iters=bc_iters, identical=make_env.get_identical(env_id), d_iters=d_iters, g_iters=g_iters)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='./results')
@click.option('--env', type=click.STRING, default='simple_spread')
@click.option('--expert_path', type=click.STRING,
              default='./results/mack_om/simple_speaker_listener/l-0.1-b-1000/seed-1/checkpoint55000-200tra.pkl')
@click.option('--seed', type=click.INT, default=1)
@click.option('--traj_limitation', type=click.INT, default=200)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)
@click.option('--disc_type', type=click.Choice(['decentralized', 'decentralized-all']), default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
@click.option('--d_iters', type=click.INT, default=1)
@click.option('--g_iters', type=click.INT, default=1)
@click.option('--ent_coef', type=click.FLOAT, default=0.0)
@click.option('--hyper_study', is_flag=True, flag_value=True)
def main(logdir, env, expert_path, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters, d_iters, g_iters, ent_coef, hyper_study):
    expert_path='./results/mack_om/'+env+'/l-0.1-b-1000/seed-'+str(1)+'/checkpoint55000-200tra-{}.pkl'.format(seed)
    print(expert_path)
    env_ids = [env]
    lrs = [0.1]
    seeds = [seed]
    batch_sizes = [1000]

    ldir = './results'

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        logdir = ldir + '/codail/' + env_id + '/' + disc_type + '/s-{}/l-{}-b-{}-d-{}-c-{}/seed-{}'.format(
              traj_limitation, lr, batch_size, dis_lr, bc_iters, seed)
        if hyper_study:
            logdir = ldir + '/codail/' + env_id + '/' + disc_type + '/hyper_study/'+'s-{}/d-{}-g-{}-c-{}/seed-{}'.format(
              traj_limitation, d_iters, g_iters, ent_coef, seed)
        else:
            d_iters = g_iters = 1
            ent_coef = 0.0
        print(logdir)
        train(logdir,
              env_id, 5e7, lr, batch_size, seed, batch_size // 250, expert_path,
              traj_limitation, ret_threshold, dis_lr, disc_type=disc_type, bc_iters=bc_iters, d_iters=d_iters, g_iters=g_iters, ent_coef=ent_coef)


if __name__ == "__main__":
    main()
