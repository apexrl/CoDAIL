import os.path as osp
import random
import time

import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from rl.acktr.utils import Scheduler, find_trainable_variables, discount_with_dones
from rl.acktr.utils import cat_entropy, mse, onehot, multionehot

from rl import logger
from rl.acktr import kfac
from rl.common import set_global_seeds, explained_variance
from irl.mack.kfac_discriminator_codail import Discriminator
# from irl.mack.kfac_discriminator_wgan import Discriminator
from irl.dataset import Dset


class Model(object):
    def __init__(self, policy, oppo_policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', identical=None):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nbatch = nenvs * nsteps
        self.num_agents = num_agents = len(ob_space)
        self.n_actions = [ac_space[k].n for k in range(self.num_agents)]
        self.op_ac_n = op_ac_n = [sum(self.n_actions) - self.n_actions[k] for k in range(self.num_agents)]
        if identical is None:
            identical = [False for _ in range(self.num_agents)]

        scale = [1 for _ in range(num_agents)]
        pointer = [i for i in range(num_agents)]
        h = 0
        for k in range(num_agents):
            if identical[k]:
                scale[h] += 1
            else:
                pointer[h] = k
                h = k
        pointer[h] = num_agents

        print(pointer)

        A, ADV, R, PG_LR = [], [], [], []
        OPPO_A = []
        for k in range(num_agents):
            if identical[k]:
                A.append(A[-1])
                ADV.append(ADV[-1])
                R.append(R[-1])
                PG_LR.append(PG_LR[-1])
                OPPO_A.append(OPPO_A[-1])
            else:
                A.append(tf.placeholder(tf.int32, [nbatch * scale[k]]))
                ADV.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                R.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                PG_LR.append(tf.placeholder(tf.float32, []))
                OPPO_A.append([tf.placeholder(tf.int32, [nbatch * scale[k]]) for _ in range(num_agents-1)])

        # A = [tf.placeholder(tf.int32, [nbatch]) for _ in range(num_agents)]
        # ADV = [tf.placeholder(tf.float32, [nbatch]) for _ in range(num_agents)]
        # R = [tf.placeholder(tf.float32, [nbatch]) for _ in range(num_agents)]
        # PG_LR = [tf.placeholder(tf.float32, []) for _ in range(num_agents)]
        # VF_LR = [tf.placeholder(tf.float32, []) for _ in range(num_agents)]
        pg_loss, entropy, vf_loss, train_loss = [], [], [], []
        self.model = step_model = []
        self.oppo_train_model = oppo_train_model = []
        self.oppo_step_model = oppo_step_model = []
        self.model2 = train_model = []
        self.pg_fisher = pg_fisher_loss = []
        self.op_fisher = op_fisher_loss = []
        self.logits = logits = []
        sample_net = []
        self.vf_fisher = vf_fisher_loss = []
        self.joint_fisher = joint_fisher_loss = []
        self.lld = lld = []
        self.oppo_lld = oppo_lld = []

        for k in range(num_agents):
            if identical[k]:
                step_model.append(step_model[-1])
                train_model.append(train_model[-1])
            else: # \pi(a^i | s, a^-i)
                oppo_train_model.append(oppo_policy(sess, k, ob_space[k], op_ac_n[k], ob_space, ac_space, 
                                        nenvs * scale[k], nsteps, nstack, reuse=False, name='%d' % k))
                oppo_step_model.append(oppo_policy(sess, k, ob_space[k], op_ac_n[k], ob_space, ac_space, 
                                        nenvs, 1, nstack, reuse=True, name='%d' % k))

                step_model.append(policy(sess, oppo_step_model[-1], ob_space[k], ac_space[k], op_ac_n[k], 
                    ob_space, ac_space, nenvs, 1, nstack, reuse=False, name='%d' % k))
                train_model.append(policy(sess, oppo_train_model[-1], ob_space[k], ac_space[k], op_ac_n[k], 
                    ob_space, ac_space, nenvs * scale[k], nsteps, nstack, reuse=True, name='%d' % k))

            logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_model[k].pi, labels=A[k])
            lld.append(tf.reduce_mean(logpac))
            logits.append(train_model[k].pi)

            oppo_logpac = []
            for op in range(num_agents-1):
                oppo_logpac.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=oppo_train_model[k].pi[op], labels=OPPO_A[k][op]))
            oppo_logpac = sum(oppo_logpac)
            oppo_lld.append(tf.reduce_mean(oppo_logpac))

            ##training loss
            pg_loss.append(tf.reduce_mean(ADV[k] * logpac))
            entropy.append(tf.reduce_mean(cat_entropy(train_model[k].pi)))
            pg_loss[k] = pg_loss[k] - ent_coef * entropy[k]
            vf_loss.append(tf.reduce_mean(mse(tf.squeeze(train_model[k].vf), R[k])))
            train_loss.append(pg_loss[k] + vf_coef * vf_loss[k])

            ##Fisher loss construction
            op_fisher_loss.append(-tf.reduce_mean(oppo_logpac))
            pg_fisher_loss.append(-tf.reduce_mean(logpac))
            sample_net.append(train_model[k].vf + tf.random_normal(tf.shape(train_model[k].vf)))
            vf_fisher_loss.append(-vf_fisher_coef * tf.reduce_mean(
                tf.pow(train_model[k].vf - tf.stop_gradient(sample_net[k]), 2)))
            joint_fisher_loss.append(pg_fisher_loss[k] + vf_fisher_loss[k])

        self.policy_params = [] # [find_trainable_variables("policy_%d" % k) for k in range(num_agents)]
        self.value_params = [] # [find_trainable_variables('value_%d' % k) for k in range(num_agents)]

        self.oppo_policy_params = [] # [find_trainable_variables("oppo_policy_%d" % k) for k in range(num_agents)]

        for k in range(num_agents):
            if identical[k]:
                self.policy_params.append(self.policy_params[-1])
                self.value_params.append(self.value_params[-1])
            else:
                self.policy_params.append(find_trainable_variables("policy_%d" % k))
                self.value_params.append(find_trainable_variables("value_%d" % k))
                self.oppo_policy_params.append(find_trainable_variables("oppo_%d" % k))

        self.params = params = [a + b for a, b in zip(self.policy_params, self.value_params)]
        self.oppo_params = oppo_params = self.oppo_policy_params
        params_flat = []
        for k in range(num_agents):
            params_flat.extend(params[k])
            params_flat.extend(oppo_params[k])

        self.grads_check = grads = [
            tf.gradients(train_loss[k], params[k]) for k in range(num_agents)
        ]
        clone_grads = [
            tf.gradients(lld[k], params[k]) for k in range(num_agents)
        ]
        oppo_clone_grads = [
            tf.gradients(oppo_lld[k], oppo_params[k]) for k in range(num_agents)
        ]

        self.optim = optim = []
        self.clones = clones = []
        self.oppo_clones = oppo_clones = []
        update_stats_op = []
        train_op, clone_op, oppo_clone_op, q_runner = [], [], [], []

        for k in range(num_agents):
            if identical[k]:
                optim.append(optim[-1])
                train_op.append(train_op[-1])
                q_runner.append(q_runner[-1])
                clones.append(clones[-1])
                clone_op.append(clone_op[-1])
                oppo_clones.append(oppo_clones[-1])
                oppo_clone_op.append(oppo_clone_op[-1])
            else:
                with tf.variable_scope('optim_%d' % k):
                    optim.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=0, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    update_stats_op.append(optim[k].compute_and_apply_stats(joint_fisher_loss[k], var_list=params[k]))
                    train_op_, q_runner_ = optim[k].apply_gradients(list(zip(grads[k], params[k])))
                    train_op.append(train_op_)
                    q_runner.append(q_runner_)

                with tf.variable_scope('clone_%d' % k):
                    clones.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=1, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    update_stats_op.append(clones[k].compute_and_apply_stats(
                        pg_fisher_loss[k], var_list=self.policy_params[k]))
                    clone_op_, q_runner_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                    clone_op.append(clone_op_)

                with tf.variable_scope('oppo_clone_%d' % k):
                    oppo_clones.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=1, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    update_stats_op.append(oppo_clones[k].compute_and_apply_stats(
                        op_fisher_loss[k], var_list=self.oppo_policy_params[k]))
                    oppo_clone_op_, q_runner_ = oppo_clones[k].apply_gradients(list(zip(oppo_clone_grads[k], self.oppo_policy_params[k])))
                    oppo_clone_op.append(oppo_clone_op_)

        update_stats_op = tf.group(*update_stats_op)
        train_ops = train_op
        # train_op = tf.group(*train_op)
        clone_ops = clone_op
        # clone_op = tf.group(*clone_op)
        oppo_clone_ops = oppo_clone_op

        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.oppo_clone_lr = self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = [rewards[k] - values[k] for k in range(num_agents)]
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            ob = np.concatenate(obs, axis=1)

            td_map = {}
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                if num_agents > 1:
                    action_v = []
                    for j in range(k, pointer[k]):
                        action_v.append(np.concatenate([multionehot(actions[i], self.n_actions[i])
                                                   for i in range(num_agents) if i != k], axis=1))
                    action_v = np.concatenate(action_v, axis=0)
                    new_map.update({train_model[k].A_v: action_v})
                    td_map.update({train_model[k].A_v: action_v})

                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].X_v: np.concatenate([ob.copy() for j in range(k, pointer[k])], axis=0),
                    train_model[k].oppo_policy.X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    ADV[k]: np.concatenate([advs[j] for j in range(k, pointer[k])], axis=0),
                    R[k]: np.concatenate([rewards[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                sess.run(train_ops[k], feed_dict=new_map)
                td_map.update(new_map)

                if states[k] != []:
                    td_map[train_model[k].S] = states
                    td_map[train_model[k].M] = masks

            policy_loss, value_loss, policy_entropy = sess.run(
                [pg_loss, vf_loss, entropy],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def clone(obs, actions):
            td_map = {}
            cur_lr = self.clone_lr.value()
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].oppo_policy.X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                sess.run(clone_ops[k], feed_dict=new_map)
                td_map.update(new_map)
            lld_loss = sess.run([lld], td_map)
            return lld_loss

        def opponent_clone(obs, actions):
            td_map = {}
            cur_lr = self.oppo_clone_lr.value()
            for k in range(num_agents):
                oppo_actions = [actions[i] for i in range(num_agents) if i!=k]
                new_map = {
                    oppo_train_model[k].X: obs[k],
                    PG_LR[k]: cur_lr / float(scale[k])
                }
                new_map.update(zip(OPPO_A[k], oppo_actions))
                td_map.update(new_map)
                sess.run(oppo_clone_ops[k], new_map)
            oppo_lld_loss = sess.run([oppo_lld], td_map)
            return oppo_lld_loss[0]

        def save(save_path):
            ps = sess.run(params_flat)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params_flat, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.clone = clone
        self.opponent_clone = opponent_clone
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model

        def step(ob, av, *_args, **_kwargs):
            a, v, s = [], [], []
            obs = np.concatenate(ob, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([multionehot(av[i], self.n_actions[i])
                                      for i in range(num_agents) if i != k], axis=1)
                a_, v_, s_ = step_model[k].step(ob[k], obs, a_v)
                a.append(a_)
                v.append(v_)
                s.append(s_)
            return a, v, s

        self.step = step

        def value(ob, av):
            v = []
            obs = np.concatenate(ob, axis=1)
            for k in range(num_agents):
                if num_agents > 1:
                    a_v = np.concatenate([multionehot(av[i], self.n_actions[i])
                                          for i in range(num_agents) if i != k], axis=1)
                else:
                    a_v = None
                v_ = step_model[k].value(ob[k], obs, a_v)
                v.append(v_)
            return v

        self.value = value
        self.initial_state = [step_model[k].initial_state for k in range(num_agents)]


class Runner(object):
    def __init__(self, env, model, discriminator, nsteps, nstack, gamma, lam, disc_type):
        self.env = env
        self.model = model
        self.discriminator = discriminator
        self.disc_type = disc_type
        self.num_agents = len(env.observation_space)
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = [
            (nenv * nsteps, nstack * env.observation_space[k].shape[0]) for k in range(self.num_agents)
        ]
        self.obs = [
            np.zeros((nenv, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ]
        self.actions = [np.zeros((nenv, )) for _ in range(self.num_agents)]
        obs = env.reset() # s_0
        self.update_obs(obs) # self.obs = obs = s_0
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.n_actions = [env.action_space[k].n for k in range(self.num_agents)]
        self.dones = [np.array([False for _ in range(nenv)]) for k in range(self.num_agents)]

    def update_obs(self, obs):
        # TODO: Potentially useful for stacking.
        self.obs = obs
        # for k in range(self.num_agents):
        #     ob = np.roll(self.obs[k], shift=-1, axis=1)
        #     ob[:, -1] = obs[:, 0]
        #     self.obs[k] = ob

        # self.obs = [np.roll(ob, shift=-1, axis=3) for ob in self.obs]
        # self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        # mb_episode_r = [[] for _ in range(self.num_agents)]

        mb_obs = [[] for _ in range(self.num_agents)]
        mb_true_rewards = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = [[] for _ in range(self.num_agents)]
        mb_masks = [[] for _ in range(self.num_agents)]
        mb_states = self.states
        mul_oppo = [None for _ in range(self.num_agents)]
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.actions) # self.actions = a_{t-1}, actions=a_t, self.obs=s_t
            self.actions = actions # self.actions = a_t
            mul = [self.actions[k] for k in range(self.num_agents)] # mul = a_{t-1}^n
            for k in range(self.num_agents):
                mul_oppo[k] = self.model.oppo_step_model[k].step(self.obs[k], np.concatenate(self.obs, axis=1), None) 
                mul_oppo[k] = mul_oppo[k][:k] + [mul[k]] + mul_oppo[k][k:]
                mul_oppo[k] = [multionehot(mul_oppo[k][i], self.n_actions[i]) for i in range(self.num_agents)] # mul = a_{t-1}^n
            if self.disc_type == 'decentralized':
                rewards = [np.squeeze(self.discriminator[k].get_reward(
                    self.obs[k], np.concatenate(mul_oppo[k], axis=1)))
                    for k in range(self.num_agents)] # rewards = r_{t-1} = r(s_t, a_{t}^n)
            elif self.disc_type == 'decentralized-all':
                rewards = [np.squeeze(self.discriminator[k].get_reward(
                    np.concatenate(self.obs, axis=1), np.concatenate(mul_oppo[k], axis=1)))
                    for k in range(self.num_agents)]
            else:
                assert False

            for k in range(self.num_agents):
                mb_obs[k].append(np.copy(self.obs[k])) # obs at t
                mb_actions[k].append(actions[k]) # actions at t
                mb_values[k].append(values[k]) # values at t
                mb_dones[k].append(self.dones[k]) # dones at t-1
                mb_rewards[k].append(rewards[k]) # rewards at t
            actions_list = []
            for i in range(self.nenv):
                actions_list.append([onehot(actions[k][i], self.n_actions[k]) for k in range(self.num_agents)])
            obs, true_rewards, dones, _ = self.env.step(actions_list) # obs = s_{t+1}
            self.states = states
            self.dones = dones
            for k in range(self.num_agents):
                for ni, done in enumerate(dones[k]):
                    if done:
                        self.obs[k][ni] = self.obs[k][ni] * 0.0
            self.update_obs(obs) # self.obs = s_t
            for k in range(self.num_agents):
                mb_true_rewards[k].append(true_rewards[k])
        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k]) # dones at t
        print(len(mb_obs[0]))
        print(len(mb_dones[0]))

        # batch of steps to batch of rollouts
        # print(mb_rewards[0])
        for k in range(self.num_agents):
            # mb_episode_r[k] = np.sum(mb_rewards[k]) / np.shape(mb_rewards[k])[-1]
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.int32).swapaxes(1, 0)
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]

        # last_values = self.model.value(self.obs, self.actions)
        #
        # mb_advs = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        # mb_returns = [[] for _ in range(self.num_agents)]
        #
        # lastgaelam = 0.0
        # for k in range(self.num_agents):
        #     for t in reversed(range(self.nsteps)):
        #         if t == self.nsteps - 1:
        #             nextnonterminal = 1.0 - self.dones[k]
        #             nextvalues = last_values[k]
        #         else:
        #             nextnonterminal = 1.0 - mb_dones[k][:, t + 1]
        #             nextvalues = mb_values[k][:, t + 1]
        #         delta = mb_rewards[k][:, t] + self.gamma * nextvalues * nextnonterminal - mb_values[k][:, t]
        #         mb_advs[k][:, t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        #     mb_returns[k] = mb_advs[k] + mb_values[k]
        #     mb_returns[k] = mb_returns[k].flatten()
        #     mb_masks[k] = mb_masks[k].flatten()
        #     mb_values[k] = mb_values[k].flatten()
        #     mb_actions[k] = mb_actions[k].flatten()

        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_true_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        last_values = self.model.value(self.obs, self.actions)
        # discount/bootstrap off value fn
        for k in range(self.num_agents):
            for n, (rewards, true_rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_true_rewards[k], mb_dones[k], last_values[k].tolist())):
                rewards = rewards.tolist()
                dones = dones.tolist()
                true_rewards = true_rewards.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                    true_rewards = discount_with_dones(true_rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                    true_rewards = discount_with_dones(true_rewards, dones, self.gamma)
                mb_returns[k][n] = rewards
                mb_true_returns[k][n] = true_rewards

        for k in range(self.num_agents):
            mb_returns[k] = mb_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()
            mb_actions[k] = mb_actions[k].flatten()

        mh_actions = [multionehot(mb_actions[k], self.n_actions[k]) for k in range(self.num_agents)]
        mb_all_obs = np.concatenate(mb_obs, axis=1)
        mh_all_actions = np.concatenate(mh_actions, axis=1)
        return mb_obs, mb_states, mb_returns, mb_masks, mb_actions,\
               mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns#, mb_episode_r


def learn(policy, oppo_policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=50, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None, d_iters=1, g_iters=1):
    tf.reset_default_graph()
    set_global_seeds(seed)
    buffer = None

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_agents = (len(ob_space))
    make_model = lambda: Model(policy, oppo_policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if disc_type == 'decentralized':
        discriminator = [
            Discriminator(model.sess, ob_space, ac_space, nstack, k, disc_type=disc_type,
                          scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                          total_steps=total_timesteps // (nprocs * nsteps),
                          lr_rate=dis_lr) for k in range(num_agents)
        ]
    elif disc_type == 'dentralized-all':
        discriminator = [
            Discriminator(model.sess, ob_space, ac_space, nstack, k, disc_type=disc_type,
                          scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                          total_steps=total_timesteps // (nprocs * nsteps),
                          lr_rate=dis_lr) for k in range(num_agents)
        ]
    else:
        assert False
    tf.global_variables_initializer().run(session=model.sess)
    runner = Runner(env, model, discriminator, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, disc_type=disc_type)
    nbatch = nenvs * nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    # enqueue_threads = [q_runner.create_threads(model.sess, coord=coord, start=True) for q_runner in model.q_runner]
    for _ in range(bc_iters):
        e_obs, e_actions, _, _ = expert.get_next_batch(nenvs * nsteps)
        e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
        lld_loss = model.clone(e_obs, e_a)
        oppo_lld_loss = model.opponent_clone(e_obs, e_a)
        # print(lld_loss)

    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values, all_obs,\
        mh_actions, mh_all_actions, mh_rewards, mh_true_rewards, mh_true_returns = runner.run()#, mh_episode_r = runner.run()
        oppo_lld_loss = model.opponent_clone(obs, actions)

        # d_iters = 1
        g_loss, e_loss = np.zeros((num_agents, d_iters)), np.zeros((num_agents, d_iters))
        idx = 0
        idxs = np.arange(len(all_obs))
        random.shuffle(idxs)
        all_obs = all_obs[idxs]
        mh_actions = [mh_actions[k][idxs] for k in range(num_agents)]
        mh_obs = [obs[k][idxs] for k in range(num_agents)]
        mh_values = [values[k][idxs] for k in range(num_agents)]

        if buffer:
            buffer.update(mh_obs, mh_actions, None, all_obs, mh_values)
        else:
            buffer = Dset(mh_obs, mh_actions, None, all_obs, mh_values, randomize=True, num_agents=num_agents)

        d_minibatch = nenvs * nsteps

        for d_iter in range(d_iters):
            e_obs, e_actions, e_all_obs, _ = expert.get_next_batch(d_minibatch)
            g_obs, g_actions, g_all_obs, _ = buffer.get_next_batch(batch_size=d_minibatch)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    g_loss[k, d_iter], e_loss[k, d_iter], _, _ = discriminator[k].train(
                        g_obs[k],
                        np.concatenate(g_actions, axis=1),
                        e_obs[k],
                        np.concatenate(e_actions, axis=1) 
                    )
            elif disc_type == 'decentralized-all':
                for k in range(num_agents):
                    g_loss[k, d_iter], e_loss[k, d_iter], _, _ = discriminator[k].train(
                        g_all_obs,
                        np.concatenate(g_actions, axis=1),
                        e_all_obs,
                        np.concatenate(e_actions, axis=1))
            else:
                assert False
            idx += 1

        if update > 10:
            for g_iter in range(g_iters):
                policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)]
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)

            for k in range(model.num_agents):
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                # logger.record_tabular("episode_reward %d" % k, float(mh_episode_r[k]))
                if update > 10:
                    logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                    logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                    logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
                    try:
                        logger.record_tabular('pearson %d' % k, float(
                            pearsonr(rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('reward %d' % k, float(np.mean(rewards[k])))
                        logger.record_tabular('spearman %d' % k, float(
                            spearmanr(rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular("oppo_lld_loss %d" % k, float(oppo_lld_loss[k]))
                    except:
                        pass
            # logger.record_tabular("episode_sum_reward", float(np.sum(mh_episode_r[k])))
            if update > 10 and env_id == 'simple_tag':
                try:
                    logger.record_tabular('in_pearson_0_2', float(
                        pearsonr(rewards[0].flatten(), rewards[2].flatten())[0]))
                    logger.record_tabular('in_pearson_1_2', float(
                        pearsonr(rewards[1].flatten(), rewards[2].flatten())[0]))
                    logger.record_tabular('in_spearman_0_2', float(
                        spearmanr(rewards[0].flatten(), rewards[2].flatten())[0]))
                    logger.record_tabular('in_spearman_1_2', float(
                        spearmanr(rewards[1].flatten(), rewards[2].flatten())[0]))
                except:
                    pass

            g_loss_m = np.mean(g_loss, axis=1)
            e_loss_m = np.mean(e_loss, axis=1)
            # g_loss_gp_m = np.mean(g_loss_gp, axis=1)
            # e_loss_gp_m = np.mean(e_loss_gp, axis=1)
            for k in range(num_agents):
                logger.record_tabular("g_loss %d" % k, g_loss_m[k])
                logger.record_tabular("e_loss %d" % k, e_loss_m[k])
                # logger.record_tabular("g_loss_gp %d" % k, g_loss_gp_m[k])
                # logger.record_tabular("e_loss_gp %d" % k, e_loss_gp_m[k])

            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'm_%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update))
                    discriminator[k].save(savepath)
            elif disc_type == 'decentralized-all':
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update))
                    discriminator[k].save(savepath)
    coord.request_stop()
    # coord.join(enqueue_threads)
    env.close()
