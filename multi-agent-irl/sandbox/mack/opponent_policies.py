import numpy as np
import tensorflow as tf

import rl.common.tf_util as U
from rl.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div


class CategoricalPolicy(object):
    def __init__(self, sess, agent_id, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        self.agent_id = agent_id
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space
        actions = [tf.placeholder(tf.int32, (nbatch)) for _ in range(len(ob_spaces)-1)]
        all_ac_shape = (nbatch, (sum([ac.n for ac in ac_spaces]) - nact) * nstack)
        obs_x = tf.placeholder(tf.float32, ob_shape)  # obs
        X = obs_x
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('oppo_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = []
            for k in range(len(ob_spaces)):
                if k == agent_id:
                    continue
                pi.append(fc(h2, 'pi_%d'%k, ac_spaces[k].n, act=lambda x: x))
        self.log_prob = [-tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi[i], labels=actions[i]) for i in range(len(pi))]
        a0 = [sample(_) for _ in pi]
        self.initial_state = []  # not stateful

        def step_log_prob(ob, acts_n):
            acts = [acts_n[i] for i in range(len(acts_n)) if i!=self.agent_id]
            feed_dict = {X:ob}
            feed_dict.update(zip(actions, acts))
            log_prob = sess.run(self.log_prob, feed_dict)
            return log_prob.reshape([-1, 1])

        def step(ob, obs, a_v, *_args, **_kwargs):
            a = sess.run(a0, {X: ob, X_v: obs})
            return a

        self.obs_x = obs_x
        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.step_log_prob = step_log_prob
        self.step = step


class GaussianPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        self.agent_id = agent_id
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        obs_x = tf.placeholder(tf.float32, ob_shape)  # obs
        X = obs_x
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('oppo_policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            h2 = fc(h1, 'fc2', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            pi = []
            for k in range(len(ob_spaces)):
                if k == agent_id:
                    continue
                pi.append(fc(h2, 'pi%d'%k, ac_spaces[k], act=lambda x: x, init_scale=0.01))

        with tf.variable_scope('oppo_{}'.format(name), reuse=reuse):
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            logstd = tf.expand_dims(logstd, 0)
            std = tf.exp(logstd)
            std = tf.tile(std, [nbatch, 1])

        a0 = pi + tf.random_normal(tf.shape(std), 0.0, 1.0) * std

        self.initial_state = []  # not stateful

        def step(ob, obs, *_args, **_kwargs):
            a = sess.run(a0, {X: ob, X_v: obs})
            return a

        self.obs_x = obs_x
        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.std = std
        self.logstd = logstd
        self.step = step
        self.mean_std = tf.concat([pi, std], axis=1)


class MultiCategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        self.agent_id = agent_id
        nbins = 11
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        obs_x = tf.placeholder(tf.float32, ob_shape)  # obs
        X = obs_x
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('oppo_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = []
            for k in range(len(ob_spaces)):
                if k == agent_id:
                    continue
                pi.append(fc(h2, 'pi%d'%k, ac_spaces[k] * nbins, act=lambda x: x))

        pi = tf.reshape(pi, [nbatch, nact, nbins])
        a0 = sample(pi, axis=2)
        self.initial_state = []  # not stateful

        def step(ob, obs, *_args, **_kwargs):
            a = sess.run(a0, {X: ob, X_v: obs})
            return a

        def transform(a):
            # transform from [0, 9] to [-0.8, 0.8]
            a = np.array(a, dtype=np.float32)
            a = (a - (nbins - 1) / 2) / (nbins - 1) * 2.0
            return a

        self.obs_x = obs_x
        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.step = step
