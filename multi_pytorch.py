import os
import logging
# import multiprocessing as mp
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import a2c_torch
import env
import load_trace
from tensorboardX import SummaryWriter


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 8
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
NN_MODEL = './results/pretrain_linear_reward.ckpt'
# NN_MODEL = None



def convert_torch(variable, dtype=np.float32):
    if variable.dtype != dtype:
        variable = variable.astype(dtype)

    return torch.from_numpy(variable)


def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    # run test script
    os.system('python rl_test_pytorch.py ' + nn_model)
    
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()
    
    return rewards_mean


def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    write_test = SummaryWriter(SUMMARY_DIR)
    with open(LOG_FILE + '_test', 'w') as test_log_file:
        actor = a2c_torch.ActorNet(s_dim=[S_INFO, S_LEN], a_dim=A_DIM, lr=ACTOR_LR_RATE)
        critic = a2c_torch.CriticNet(s_dim =[S_INFO, S_LEN], lr=CRITIC_LR_RATE)

        
        actor_optim = optim.RMSprop(actor.parameters(), lr=ACTOR_LR_RATE)
        critic_optim = optim.RMSprop(critic.parameters(), lr=CRITIC_LR_RATE)
        
        epoch = 0
        while True:
            actor_net_params = actor.state_dict()
            critic_net_params = critic.state_dict()
            # print(actor_net_params)
            for i in range(NUM_AGENTS):

                net_params_queues[i].put([actor_net_params, critic_net_params])
            # print('parameters hahaha: ', actor_net_params)
            # print('parameters: ', list(actor_net_params))
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0 
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, old_pi_batch, terminal, info = exp_queues[i].get()
                # print('exp_queue')
                s_batch, a_batch, \
                    r_batch, old_pi_batch, terminal = convert_torch(np.array(s_batch)), convert_torch(np.array(a_batch)), \
                                                        convert_torch(np.array(r_batch)), convert_torch(np.array(old_pi_batch)), convert_torch(np.array(terminal))
                # actor.cuda(), critic.cuda()
                # s_batch.cuda(), a_batch.cuda(), r_batch.cuda(), old_pi_batch.cuda(), terminal.cuda()
                
                critic_loss, td_batch = critic.cal_loss(s_batch, r_batch, terminal)
                actor_loss = actor.cal_loss(s_batch, a_batch, td_batch, epoch)

                critic_loss.backward()
                actor_loss.backward()
                total_reward += np.sum(r_batch.numpy())
                total_td_loss += np.sum(td_batch.numpy())
                total_batch_len += len(r_batch.numpy())
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])
            critic_optim.step()
            actor_optim.step()
            # actor.cpu(), critic.cpu()
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            # logging.info('Epoch: ' + str(epoch) +
            #              ' TD_loss: ' + str(avg_td_loss) +
            #              ' Avg_reward: ' + str(avg_reward) +
            #              ' Avg_entropy: ' + str(avg_entropy))

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                print('Epoch = ', epoch)
                torch.save(actor.state_dict(), SUMMARY_DIR + "/actor_nn_model_ep_" +
                                       str(epoch) + ".pkl")
                torch.save(critic.state_dict(), SUMMARY_DIR + "/critic_nn_model_ep_" +
                                       str(epoch) + ".pkl")
                
                # logging.info("Model saved in file: " + save_path)
                reward_mean = testing(epoch, 
                    SUMMARY_DIR + "/actor_nn_model_ep_" + str(epoch) + ".pkl", 
                    test_log_file)
                
                print('epoch = ', epoch, 'reward = ', reward_mean)
                write_test.add_scalar('Testing/total_reward', reward_mean, epoch)
                write_test.add_scalar('Training/Entropy', avg_entropy, epoch)
                write_test.add_scalar('Training/TD_Error', avg_td_loss, epoch)

                write_test.flush()
                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: avg_td_loss,
                #     summary_vars[1]: reward_mean,
                #     summary_vars[2]: avg_entropy
                # })

                # writer.add_summary(summary_str, epoch)
                # writer.flush()
        


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        actor = a2c_torch.ActorNet(s_dim=[S_INFO, S_LEN], a_dim=A_DIM, lr=ACTOR_LR_RATE)
        critic = a2c_torch.CriticNet(s_dim =[S_INFO, S_LEN], lr=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.load_state_dict(actor_net_params)
        critic.load_state_dict(critic_net_params)
        

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        old_pi_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            # -- log scale reward --
            # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

            # reward = log_bit_rate \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            # reward = HD_REWARD[bit_rate] \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector
            _, _, action_prob = actor.get_actor_out(convert_torch(np.reshape(state, (1, S_INFO, S_LEN))))
            action_prob = action_prob.numpy()
            # print(action_prob)
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a2c_torch.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               old_pi_batch[1:],
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.load_state_dict(actor_net_params)
                critic.load_state_dict(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del old_pi_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                old_pi_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)
                old_pi_batch.append(action_prob)


def main():

    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
