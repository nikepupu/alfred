import numpy as np
from env.thor_env import ThorEnv
from tqdm import tqdm
import os
import json
import argparse
from models.model.action_model import A3CLSTMNSingleAgent, a3c_loss
import pickle
import torch
import torch.nn.functional as F
import models.nn.vnn as vnn
import warnings
from gen.utils.py_util import remove_spaces_and_lower
import revtok
import itertools
from setproctitle import setproctitle as ptitle
from shared_optim import SharedRMSprop, SharedAdam
import torch.multiprocessing as mp

def ensure_shared_grads(model, shared_model, gpu=False):
    sentinal = object()
    
    for param, shared_param in itertools.zip_longest(
        model.parameters(), shared_model.parameters(), fillvalue=sentinal
    ):
        assert sentinal is not param and sentinal is not shared_param
        if shared_param.requires_grad:
            assert param.requires_grad
            if not gpu or param.grad is None:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()
                
def train_RL(env, model, shared_model, optimizer, lang_goal, actions, gpu_id=-1, mode='train'):
    log_prob_of_actions = []
    entropy_per_agent = []
    values_per_agent = []
    log_prob_of_expert_action = []
    log_prob_of_unsafe_action = []
    eval_results = []
    rewards_per_agent = []
    a3c_gamma = 0.99
    a3c_tau = 1.0
    a3c_beta = 1e-2
    
    if gpu_id >= 0:
        hidden =  tuple((torch.zeros(1, 1, 512).cuda(), torch.zeros(1, 1, 512).cuda() ) )
    else:
        hidden =  tuple((torch.zeros(1, 1, 512), torch.zeros(1, 1, 512)) )
    
    done = False
    if mode == 'train':
        model = model.train()
    else:
        model = model.eval()
    for t in range(20):
        if done:
            break
        if gpu_id >= 0:
            inputs = torch.Tensor(env.last_event.frame.copy()).permute(2,0,1).unsqueeze(0).cuda()
        else:
            inputs = torch.Tensor(env.last_event.frame.copy()).permute(2,0,1).unsqueeze(0)

        
 
        eval_result = model(inputs, hidden, lang_goal,None)
        hidden = eval_result.get("hidden")
        logit_per_agent = eval_result["actor"][0].unsqueeze(0)
        probs_per_agent =  F.softmax(logit_per_agent, dim=1) 
        log_probs_per_agent = F.log_softmax(logit_per_agent, dim=1) 
        
        entropy_per_agent.append(
            -(log_probs_per_agent * probs_per_agent).sum().unsqueeze(0)
        )
        
        action = probs_per_agent.multinomial(num_samples=1).item()
        log_prob_of_action_per_agent = log_probs_per_agent.view(-1)[action]
        values_per_agent.append(eval_result.get("critic"))
        
        action_name = actions[action]
        
        for obj in env.last_event.metadata['objects']:
            
            if obj['name'].startswith(action_name['objectId']):
                action_name['objectId'] = obj['objectId']
            elif 'receptableObjectId' in action_name and obj['name'].startswith(action_name['receptableObjectId']):
                action_name['receptableObjectId'] = obj['objectId']

        env.step(action_name)        
    
        last_reward, done = env.get_transition_reward()
        rewards_per_agent.append(last_reward)
        log_prob_of_actions.append(log_prob_of_action_per_agent)
    
    if done:
        future_reward_est = 0
    else:
        future_reward_est = eval_result.get('critic')
    # print(len(values_per_agent))
    # print(len(rewards_per_agent))
    # print(len(log_prob_of_actions))
    # print(len(entropy_per_agent))
    print('mode: ' + mode + str(sum(rewards_per_agent)))
    if mode == 'train':
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        loss_dict = {}
        # rewards_per_agent = torch.Tensor(rewards_per_agent).cuda()
        a3c_losses = a3c_loss(
            values= values_per_agent,
            rewards= rewards_per_agent,
            log_prob_of_actions= log_prob_of_actions,
            entropies= entropy_per_agent,
            future_reward_est=future_reward_est,
            gamma=a3c_gamma,
            tau=a3c_tau,
            beta=a3c_beta,
            gpu_id=gpu_id,
            huber_delta=None,
        )
        
        policy_loss += a3c_losses["policy"]
        value_loss += a3c_losses["value"]
        entropy_loss += a3c_losses["entropy"]

        policy_loss /= len(rewards_per_agent)
        value_loss /= len(rewards_per_agent)
        entropy_loss /= len(rewards_per_agent)
        loss_dict["a3c_policy_loss"] = policy_loss
        loss_dict["a3c_value_loss"] = value_loss
        loss_dict["a3c_entropy_loss"] = entropy_loss
        full_loss = None
        last_losses = {}
        optimizer.zero_grad()
        for k, loss in loss_dict.items():
            loss = loss.squeeze()
        
            last_losses["loss/" + k] = loss.item()
            if full_loss is None:
                full_loss = loss
            elif (full_loss.is_cuda == loss.is_cuda) and (
                not full_loss.is_cuda or full_loss.get_device() == loss.get_device()
            ):
                full_loss += loss
            else:
                warnings.warn("Loss {} is on a different device!".format(k))
        full_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3, "inf")
        ensure_shared_grads(model, shared_model, gpu= gpu_id >= 0)
        optimizer.step()
    
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            model.load_state_dict(shared_model.state_dict())
    else:
        model.load_state_dict(shared_model.state_dict())
    
def numericalize(vocab, words, train=True):
        '''
        converts words to unique integers
        '''
        return vocab.word2index([w.strip().lower() for w in words], train=train)
    

def setup_scene(env, traj_data, r_idx, args, reward_type='dense'):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))

    # print goal instr
    print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    # setup task for reward
    env.set_task(traj_data, args, reward_type=reward_type)


def train(worker_num,  shared_model, optimizer, split='train', gpu_id=-1):
    ptitle(split + " Agent: {}".format(worker_num))
    torch.manual_seed(worker_num)
    tasks = os.listdir('data/full_2.1.0/' + split)
    parser = argparse.ArgumentParser()

    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--num_threads', type=int, default=1)

    # eval params
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--debug', dest='debug', action='store_true')
    args = parser.parse_args()
    env = ThorEnv()
    subtasks = ['CleanObject', 'CoolObject', 'GotoLocation', 'HeatObject', 'PickupObject', 'PutObject', 'SliceObject', 'ToggleObject']
   
    
    with open('actions.pickle', 'rb') as fp:
        actions = pickle.load(fp)
        actions = [eval(action) for action in actions]
    
    vocab = torch.load('pp.vocab')
    
        
    model = A3CLSTMNSingleAgent(num_inputs_per_agent=3, num_outputs=15, state_repr_length=512, vocab=vocab)
    
    
    if gpu_id >= 0:
        model = model.cuda()
        with torch.cuda.device(gpu_id):
            torch.cuda.manual_seed(100+worker_num)
    
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            model.load_state_dict(shared_model.state_dict())
            
          
    else:
        model.load_state_dict(shared_model.state_dict())
                  
    while True:
        for task in tqdm(tasks):
            task_path =  os.path.join('data/full_2.1.0/' + split, task)
            trials = os.listdir(task_path)
            
            for trial in trials:
                
                path = os.path.join('data/full_2.1.0/' + split, task, trial, 'traj_data.json')
                
                with open(path) as f:
                    traj_data = json.load(f)
                
                reward_type = 'dense'
                scene_num = traj_data['scene']['scene_num']
                object_poses = traj_data['scene']['object_poses']
                dirty_and_empty = traj_data['scene']['dirty_and_empty']
                object_toggles = traj_data['scene']['object_toggles']

                scene_name = 'FloorPlan%d' % scene_num
                
                for r_idx in range(len(traj_data['turk_annotations']['anns'])):
                    for eval_idx in range(len(traj_data['turk_annotations']['anns'][r_idx]['high_descs'])):
                        subtask = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
                        
                        if not subtask == subtasks[4]:
                            continue
                        
                        traj = {}
                        traj['ann']  = {
                            'goal': revtok.tokenize(remove_spaces_and_lower(traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]))
                        }
                        traj['num'] = {}
                        traj['num']['lang_goal'] = numericalize(vocab['word'], traj['ann']['goal'], train=True)
                
                        lang_goal = traj['num']['lang_goal']
                        
                        env.reset(scene_name)
                        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

                        # initialize to start position
                        env.step(dict(traj_data['scene']['init_action']))

                        # print goal instr
                        # print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

                        # setup task for reward
                        env.set_task(traj_data, args, reward_type=reward_type)
                
                        setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

                        # expert demonstration to reach eval_idx-1
                        expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]

                        # subgoal info
                        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
                        subgoal_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]
                    
                        # print subgoal info
                        # print("Evaluating: Subgoal %s (%d)\nInstr: %s" % ( subgoal_action, eval_idx, subgoal_instr))
                        # previous action for teacher-forcing during expert execution (None is used for initialization)
                        prev_action = None

                        done, subgoal_success = False, False
                        fails = 0
                        t = 0
                        reward = 0
                        
                    
                        while not done:
                            # break if max_steps reached
                            if t >= args.max_steps + len(expert_init_actions):
                                break

                            # extract visual feats
                    

                            # expert teacher-forcing upto subgoal
                            if t < len(expert_init_actions):
                                # get expert action
                                action = expert_init_actions[t]
                                subgoal_completed = traj_data['plan']['low_actions'][t+1]['high_idx'] != traj_data['plan']['low_actions'][t]['high_idx']
                                compressed_mask = action['args']['mask'] if 'mask' in action['args'] else None
                                mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None
                                
                                # execute expert action
                                success, _, _, err, _ = env.va_interact(action['action'], interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                                if not success:
                                    print ("expert initialization failed")
                                    break

                                # update transition reward
                                _, _ = env.get_transition_reward()
                            else:
                                ## this will be our model
                
                                train_RL(env, model, shared_model, optimizer, lang_goal, actions, gpu_id=gpu_id, mode=split)
                                
                                break
                            t += 1
                    
if __name__ == '__main__':
    import ctypes
    import time
    vocab = torch.load('pp.vocab')
    
    shared_model = A3CLSTMNSingleAgent(num_inputs_per_agent=3, num_outputs=15, state_repr_length=512, vocab=vocab)
    shared_model.share_memory()
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=1e-5)
    optimizer_state = None
    optimizer = SharedAdam(
                filter(lambda param: param.requires_grad, list(shared_model.parameters()) ),
                lr=1e-5,
                amsgrad=False,
                saved_state=optimizer_state,
            )
    processes = []
    
    mp = mp.get_context("spawn")
    valid_total_ep = mp.Value( ctypes.c_int32, 0 )
    lock = mp.Lock()
    # train(0,shared_model,optimizer, gpu_id=0)
    for i in range(2):
        p = mp.Process(
                target=train,
                args=(
                    i,
                    shared_model,
                    optimizer,
                    'train',
                    0
                ),
            )

        p.start()
        processes.append(p)
        time.sleep(0.2)

   
    p = mp.Process(
            target=train,
            args=(
                i,
                shared_model,
                optimizer,
                'valid_seen',
                0
            ),
        )

    p.start()
    processes.append(p)
    time.sleep(0.2)
    
    
    p = mp.Process(
        target=train,
        args=(
            i,
            shared_model,
            optimizer,
            'valid_unseen',
            0
        ),
    )

    p.start()
    processes.append(p)
    time.sleep(0.2)

