import gym
import click
import multiagent
import time
import tensorflow as tf
import make_env
import numpy as np
from rl.common.misc_util import set_global_seeds
from sandbox.mack.acktr_disc import Model, onehot
from sandbox.mack.acktr_disc_om import Model as Model_om
from sandbox.mack.policies import CategoricalPolicy
from sandbox.mack.policies_om import CategoricalPolicy as CategoricalPolicy_om
from sandbox.mack.opponent_policies import CategoricalPolicy as oppo_CategoricalPolicy
from rl import bench
import imageio
import seaborn as sns
import pickle as pkl
from sklearn.neighbors.kde import KernelDensity


@click.command()
@click.option('--image', is_flag=True, flag_value=True)
@click.option('--vis_dis', is_flag=True, flag_value=True)
@click.option('--hyper_study', is_flag=True, flag_value=True)
@click.option('--d', type=click.INT, default=1)
@click.option('--g', type=click.INT, default=1)
@click.option('--ent_coef', type=click.FLOAT, default=0)
@click.option('--kl', is_flag=True, flag_value=True)
@click.option('--env', type=click.STRING, default='simple_speaker_listener')
@click.option('--algo', type=click.STRING, default='mack')
@click.option('--disc_type', type=click.STRING, default='decentralized')
# @click.option('--path', type=click.STRING, default='./results/mack/simple_speaker_listener/l-0.1-b-1000/seed-1/checkpoint07000')
@click.option('--seed', type=click.INT, default=1)
@click.option('--epoch', type=click.INT, default=55000)
@click.option('--sample', is_flag=True, flag_value=True)
@click.option('--all_exp', is_flag=True, flag_value=True)
@click.option('--curve', is_flag=True, flag_value=True)
@click.option('--num_trajs', type=click.INT, default=100)
@click.option('--traj_limitation', type=click.INT, default=200)
def render(env, algo, seed, image, vis_dis, hyper_study, epoch, sample, num_trajs, disc_type, d, g, ent_coef, traj_limitation, all_exp, curve, kl):
    env_name = env

    env_id = env

    def create_env(seed):
        env = make_env.make_env(env_id)
        env.seed(seed)
        # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
        set_global_seeds(seed)
        return env

    env = create_env(0)
    n_agents = len(env.action_space)

    ob_space = env.observation_space
    ac_space = env.action_space

    #print('observation space')
    #print(ob_space)
    #print('action space')
    #print(ac_space)

    n_actions = [action.n for action in ac_space]

    result_dir = './results'
    algos = [algo]
    seeds = [seed]
    epochs = [epoch]
    if kl and algo!= "mack_om":
        algos=['mack_om', 'codail', 'ncdail', 'gail', 'airl', 'random']
        # algos=['codail', 'ncdail', 'gail', 'airl', 'random']
        seeds=[1]
        with open('./kl_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'.csv', 'w') as f:
            f.write("env_name,algo,seed,epoch,num_trajs,traj_limitation,total_kl")
            for k in range(n_agents):
                if env.world.agents[k].movable:
                    f.write(',kl_agent{}'.format(k))
            f.write('\n')
    if hyper_study:
        algos=['codail']
        seeds=[1,2,3,4,5]
        with open('./hyper_result_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'_'+str(d)+'-'+str(g)+'-'+str(ent_coef)+'.csv', 'w') as f:
            #f.write("env: {}\n".format(env_name))
            f.write("env_name,algo,seed,epoch,traj_limitation,d,g,ent_coef,mean_sum_reward")
            for k in range(n_agents):
                f.write(',agent{}'.format(k))
            f.write('\n')
    if all_exp:
        algos=['mack_om', 'codail', 'ncdail', 'gail', 'airl', 'random']
        seeds=[1,2,3,4,5]
        with open('./all_result_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'.csv', 'w') as f:
            #f.write("env: {}\n".format(env_name))
            f.write("env_name,algo,seed,epoch,traj_limitation,mean_sum_reward")
            for k in range(n_agents):
                f.write(',agent{}'.format(k))
            f.write('\n')
    if curve:
        # algos=['mack_om', 'codail', 'ncdail', 'gail', 'airl', 'random']
        algos=['codail', 'ncdail', 'gail', 'airl', 'random']
        seeds=[1]
        epochs = [1] + [_ for _ in range(50, epoch, 50)]
        with open('./curve_data_'+env_name+'_'+str(traj_limitation)+'.csv', 'w') as f:
            f.write("algo,seed,epoch,episodes,num_trajs,traj_limitation,mean_sum_reward")
            for k in range(n_agents):
                f.write(',agent{}'.format(k))
            f.write('\n')

    for epoch in epochs:
        for seed in seeds:
            for algo in algos:
                env = create_env(seed)
                tf.reset_default_graph()
                # path = './results/mack/simple_speaker_listener/l-0.1-b-1000/seed-1/checkpoint04000'
                # path = './results/gail/simple_speaker_listener/decentralized/s-200/l-0.1-b-1000-d-0.1-c-500/seed-1/m_04500'
                path = './results/'+algo+'/'+env_name+'/'
                if algo in ['mack', 'mppo', 'mack_om']:
                    path += '/l-0.1-b-1000/seed-'+str(1)+'/checkpoint' + '%05d'%epoch
                elif algo in ['codail', 'ncdail', 'gail']:
                    path += disc_type + '/' + 's-' + str(traj_limitation) +'/l-0.1-b-1000-d-0.1-c-500/seed-'+str(seed)+'/m_' + '%05d'%epoch
                elif algo in ['airl']:
                    path += disc_type + '/' + 's-' + str(traj_limitation) +'/l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-'+str(seed)+'/m_' + '%05d'%epoch

                if hyper_study:
                    path = './results/'+algo+'/'+env_name+'/' + disc_type + '/hyper_study/' + 's-' + str(traj_limitation) + '/d-{}-g-{}-c-{:.1f}'.format(d,g,ent_coef)+'/seed-'+str(seed)+'/m_' + '%05d'%epoch

                print(path)

                make_model = lambda: Model(
                    CategoricalPolicy, ob_space, ac_space, 1, total_timesteps=1e7, nprocs=2, nsteps=500,
                    nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.01, max_grad_norm=0.5, kfac_clip=0.001,
                    lrschedule='linear', identical=make_env.get_identical(env_id), use_kfac=False)
                if algo in ['mack_om', 'codail']:
                    make_model = lambda: Model_om(
                        CategoricalPolicy_om, oppo_CategoricalPolicy, ob_space, ac_space, 1, total_timesteps=1e7, nprocs=2, nsteps=500,
                        nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.01, max_grad_norm=0.5, kfac_clip=0.001,
                        lrschedule='linear', identical=make_env.get_identical(env_id), use_kfac=False)

                model = make_model()
                if algo != "random":
                    print("load model from", path)
                    model.load(path)

                images = []
                sample_trajs = []
                # num_trajs = 100
                max_steps = 50
                avg_ret = [[] for _ in range(n_agents)]

                episode_r_n_all = []
                episode_r_n_sum = []

                all_pos = [[] for k in range(n_agents)]
                for i in range(num_trajs):
                    all_ob, all_agent_ob, all_ac, all_rew, ep_ret = [], [], [], [], [0 for k in range(n_agents)]
                    for k in range(n_agents):
                        all_ob.append([])
                        all_ac.append([])
                        all_rew.append([])
                    obs = env.reset()
                    obs = [ob[None, :] for ob in obs]
                    action = [np.zeros([1]) for _ in range(n_agents)]
                    step = 0
                    done = False
                    episode_r_n = [0.] * n_agents
                    while (not done) or (step<=2):
                        if algo != "random":
                            action, _, _ = model.step(obs, action)
                            actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
                        else:
                            actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]

                        for k in range(n_agents):
                            all_ob[k].append(obs[k])
                            all_ac[k].append(actions_list[k])
                        all_agent_ob.append(np.concatenate(obs, axis=1))
                        obs, rew, done, _ = env.step(actions_list)
                        episode_r_n = [episode_r_n[i]+rew[i] for i in range(n_agents)]
                        for k in range(n_agents):
                            all_rew[k].append(rew[k])
                            ep_ret[k] += rew[k]
                            all_pos[k].append(np.array(env.world.agents[k].state.p_pos))

                        obs = [ob[None, :] for ob in obs]
                        step += 1

                        if image:
                            img = env.render(mode='rgb_array')
                            images.append(img[0])
                            time.sleep(0.02)
                        if step == max_steps or True in done:
                            done = True
                            # step = 0
                        else:
                            done = False

                    # print("\n--- episode-{} | [sum-reward]: {}".format(i, np.sum(episode_r_n)))
                    episode_r_n_all.append(episode_r_n)
                    episode_r_n_sum.append(np.sum(episode_r_n))

                    for k in range(n_agents):
                        all_ob[k] = np.squeeze(all_ob[k])

                    all_agent_ob = np.squeeze(all_agent_ob)
                    traj_data = {
                        "ob": all_ob, "ac": all_ac, "rew": all_rew,
                        "ep_ret": ep_ret, "all_ob": all_agent_ob
                    }

                    sample_trajs.append(traj_data)
                    # print('episode', i, 'expected_return', ep_ret)

                    for k in range(n_agents):
                        avg_ret[k].append(ep_ret[k])

                print("env: {}".format(env_name))
                print("seed {}, mean sum reward in {} episodes: {}".format(seed, num_trajs, np.mean(episode_r_n_sum)))
                # print(path)
                for k in range(n_agents):                    
                    print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))
                print("\n")

                if hyper_study:
                    with open('./hyper_result_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'_'+str(d)+'-'+str(g)+'-'+str(ent_coef)+'.csv', 'a') as f:
                        f.write("{},{},{},{},{},{},{},{},{}".format(env_name,algo,seed,epoch,traj_limitation,d,g,ent_coef,np.mean(episode_r_n_sum)))
                        for k in range(n_agents):
                            f.write(',{}'.format(np.mean(avg_ret[k])))
                        f.write('\n')
                if all_exp:       
                    with open('./all_result_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'.csv', 'a') as f:
                        f.write("{},{},{},{},{},{}".format(env_name,algo,seed,epoch,traj_limitation,np.mean(episode_r_n_sum)))
                        #f.write("algo: {}\n".format(algo))
                        #f.write("seed {}, mean sum reward in {} episodes: {}\n".format(seed, num_trajs, np.mean(episode_r_n_sum)))
                        for k in range(n_agents):
                            f.write(',{}'.format(np.mean(avg_ret[k])))
                            #f.write('agent{}: {}\n'.format(k, np.mean(avg_ret[k])))
                        #f.write('--------------------\n')
                        f.write('\n')
                if curve:
                    with open('./curve_data_'+env_name+'_'+str(traj_limitation)+'.csv', 'a') as f:
                        f.write("{},{},{},{},{},{},{}".format(algo,seed,epoch,i,num_trajs,traj_limitation,np.mean(episode_r_n_sum)))
                        for k in range(n_agents):
                            f.write(',{}'.format(np.mean(avg_ret[k])))
                        f.write('\n')

                images = np.array(images)
                if sample:
                    print("saving sample trajs.")
                    pkl.dump(sample_trajs, open(path + '-%dtra-%d.pkl' % (num_trajs,seed), 'wb'))
                if image:
                    print(images.shape)
                    # imageio.mimsave(path + '.mp4', images, fps=25)
                if vis_dis or kl:
                    pos = np.concatenate([all_pos[k] for k in range(n_agents) if env.world.agents[k].movable])
                    print("There are %d points in total" % len(pos))
                    if vis_dis:
                        g = sns.jointplot(x=pos[:,0], y=pos[:,1], kind='kde', space=0, color=[0.35,0.35,0.85], xlim=(-2,2), ylim=(-2,2))
                        g.savefig('interactions_'+algo+'_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'.pdf')
                        for k in range(n_agents):
                            if not env.world.agents[k].movable:
                                continue
                            print("There are %d points in total for agent %d" % (len(pos), k))
                            g = sns.jointplot(x=np.array(all_pos[k])[:,0], y=np.array(all_pos[k])[:,1], kind='kde', space=0, color=[0.35,0.85,0.35], xlim=(-2,2), ylim=(-2,2))
                            g.savefig('interactions_'+algo+'_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'_agent'+str(k)+'.pdf')
                    if kl:
                        if algo=="mack_om": # save interactions
                            for k in range(n_agents):
                                np.save("demonstrator_interaction-{}-{}-{}.npy".format(num_trajs,seed,k), all_pos[k])
                        # compute kl
                        else:
                            demon_all_pos = []
                            for k in range(n_agents):
                                demon_all_pos.append(np.load("demonstrator_interaction-{}-{}-{}.npy".format(num_trajs,seed,k)))
                            with open('./kl_'+env_name+'_'+str(traj_limitation)+'_'+str(epoch)+'.csv', 'a') as f:
                                demon_pos = np.concatenate([demon_all_pos[k] for k in range(n_agents) if env.world.agents[k].movable])
                                total_kl = kl_divergence(pos, demon_pos)
                                f.write("{},{},{},{},{},{},{}".format(env_name,algo,seed,epoch,num_trajs,traj_limitation,total_kl))
                                for k in range(n_agents):
                                    if not env.world.agents[k].movable:
                                        continue
                                    # compute kl and save
                                    f.write(",{}".format(kl_divergence(np.array(all_pos[k]), np.array(demon_all_pos[k]))))
                                f.write("\n")
                                 

def kl_divergence(x1, x2, scale=100):
    p = kde_prob(x1,min_v=-2,max_v=2,scale=100)
    q = kde_prob(x2,min_v=-2,max_v=2,scale=100)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) / scale**2 # should be rescaled!
def kde_prob(x,min_v=-2,max_v=2,scale=100):
    # print(x)
    kde = KernelDensity(kernel="gaussian", bandwidth=(max_v-min_v)*1.0/scale).fit(list(x)) # x.shape: [None, 2]
    data = [(i*1.0/scale,j*1.0/scale) for i in range(min_v*scale,max_v*scale) for j in range(min_v*scale,max_v*scale)]
    prob = np.exp(kde.score_samples(data))+1e-4 # x.shape: [None, 1]
    return prob

if __name__ == '__main__':
    render()
