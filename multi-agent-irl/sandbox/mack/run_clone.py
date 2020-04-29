import click
import time
import tensorflow as tf
import numpy as np
import os
import os.path as osp

import make_env
import gym
import logging
from rl import bench
from rl.common.misc_util import set_global_seeds
from rl import logger
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from sandbox.mack.policies import CategoricalPolicy
from sandbox.mack.acktr_disc import Model
from irl.dataset import MADataSet

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def learn(policy, env, expert, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=4, nsteps=20,
          nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.05, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=1000, lrschedule='linear', batch_size=1024):
    tf.reset_default_graph()
    set_global_seeds(seed)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs,
                               nsteps=batch_size, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef,
                               vf_fisher_coef=vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    tstart = time.time()
    coord = tf.train.Coordinator()
    # enqueue_threads = [q_runner.create_threads(model.sess, coord=coord, start=True) for q_runner in model.q_runner]
    print("-------------------------------")
    print(total_timesteps//batch_size+1)
    print("-------------------------------")

    for update in range(total_timesteps//batch_size+1):
        e_obs, e_actions, _, _ = expert.get_next_batch(batch_size)
        e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
        nseconds = time.time() - tstart
        fps = int((update * batch_size) / nseconds)

        lld_loss = model.clone(e_obs, e_a)[0]
        # print(lld_loss)
        if update % log_interval == 0 or update == 1:
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*batch_size)
            logger.record_tabular("fps", fps)
            for k in range(model.num_agents):
                logger.record_tabular("lld_loss %d" % k, float(lld_loss[k]))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)

    coord.request_stop()

@click.command()
@click.option('--logdir', type=click.STRING, default='./results')
@click.option('--env', type=click.Choice(['simple', 'simple_speaker_listener', 'simple_crypto',
                                          'simple_push', 'simple_tag']))
@click.option('--expert_path', type=click.STRING)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=1000)
@click.option('--traj_limitation', type=click.INT, default=100)
@click.option('--lr', type=click.FLOAT, default=0.01)
def train(logdir, env, expert_path, seed, batch_size, lr, traj_limitation):
    env_id = env
    logdir = logdir + '/bc/' + env_id + '/s-{}/l-{}-b-{}/seed-{}'.format(traj_limitation, lr, batch_size, seed)
    print(logdir, env, expert_path, seed)
    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    expert = MADataSet(expert_path, ret_threshold=-10, traj_limitation=traj_limitation)

    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    env = SubprocVecEnv([create_env(i) for i in range(1)], is_multi_agent=True)

    policy_fn = CategoricalPolicy
    learn(policy_fn, env, expert, seed, int(2e7), batch_size=batch_size, lr=lr )


if __name__ == '__main__':
    train()
