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
from tensorboardX import SummaryWriter
summary_writer = SummaryWriter(log_dir='./exp')


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
                
def train_RL(env, model, shared_model, optimizer, lang_goal, actions, iter, gpu_id=-1, mode='train'):
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
    t = 0
    while t < 10:
        if done:
            break
        if gpu_id >= 0:
            inputs = torch.Tensor(env.last_event.frame.copy()).permute(2,0,1).unsqueeze(0).cuda()
        else:
            inputs = torch.Tensor(env.last_event.frame.copy()).permute(2,0,1).unsqueeze(0)

        
 
        eval_result = model(inputs, hidden, lang_goal,None)
        hidden = eval_result.get("hidden")
        logit_per_agent = eval_result["actor"][0].unsqueeze(0)
        
        # short = [a['action'] for a in actions]
        # short.sort()
        # short = set(short)
        # print(short)
        for i in range(logit_per_agent.shape[1]):
            action = actions[i]
            verb = action['action']
            if 'objectId' in action:
                noun = action['objectId']
                find_it = False
                ourObj = None
                for obj in env.last_event.metadata['objects']:
                    # print(obj['name'])
                    if obj['name'].startswith(noun):
                        find_it = True
                        ourObj = obj
                        break
                    
                if not find_it:
                    logit_per_agent[0][i] = 0.0
                    
                elif verb == 'PickupObject' and not ourObj['pickupable']:
                    logit_per_agent[0][i] = 0.0
                    
                elif verb == 'CloseObject' and  not ourObj['openable'] and not ourObj['isOpen']:
                    logit_per_agent[0][i] = 0.0
                    
                elif verb == 'OpenObject' and  not ourObj['openable'] and ourObj['isOpen']:
                    logit_per_agent[0][i] = 0.0
                
                elif verb == 'ToggleObjectOff' and not ourObj['toggleable']:
                    logit_per_agent[0][i] = 0.0
                
                elif verb == 'ToggleObjectOn' and not ourObj['toggleable']:
                    logit_per_agent[0][i] = 0.0
           
                elif verb == 'SliceObject' and not ourObj['sliceable']:
                    logit_per_agent[0][i] = 0.0
                    
            if 'receptacleObjectId' in action:
                noun = action['receptacleObjectId']
                find_it = False
                ourObj = None
                for obj in env.last_event.metadata['objects']:
                    # print(obj['name'])
                    if obj['name'].startswith(noun):
                        find_it = True
                        ourObj = obj
                        break
                if not find_it:
                    logit_per_agent[0][i] = 0.0
                elif verb == 'PutObject' and not ourObj['receptacle']:
                    logit_per_agent[0][i] = 0.0
                    
            
        probs_per_agent =  F.softmax(logit_per_agent, dim=1) 
        log_probs_per_agent = F.log_softmax(logit_per_agent, dim=1) 
        
      
        
        action = probs_per_agent.multinomial(num_samples=1).item()
        log_prob_of_action_per_agent = log_probs_per_agent.view(-1)[action]
        
        action_name = actions[action]
        cnt = 0
        for k,v in action_name.items():
            if 'objectId' == k or 'receptableObjectId' == k:
                cnt += 1
                
        for obj in env.last_event.metadata['objects']:
            
            if 'objectId' in action_name and obj['name'].startswith(action_name['objectId']):
                action_name['objectId'] = obj['objectId']
                cnt -= 1
            elif 'receptableObjectId' in action_name and obj['name'].startswith(action_name['receptableObjectId']):
                action_name['receptableObjectId'] = obj['objectId']
                cnt -= 1
        # if cnt ==0:
        # print(action_name)
        env.step(action_name)        
        last_reward, done = env.get_transition_reward()
        
        rewards_per_agent.append(last_reward)
        log_prob_of_actions.append(log_prob_of_action_per_agent)
        entropy_per_agent.append(
            -(log_probs_per_agent * probs_per_agent).sum().unsqueeze(0)
        )
        values_per_agent.append(eval_result.get("critic"))
        t += 1
        # else:
        #     pass
        
        
    
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
        
    return sum(rewards_per_agent)

def train_IL(env, model, shared_model, optimizer, lang_goal, actions, iter, gpu_id=-1, mode='train'):
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
    for t in range(10):
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
        cnt = 0
        for k,v in action_name.items():
            if 'objectId' == k or 'receptableObjectId' == k:
                cnt += 1
                
        for obj in env.last_event.metadata['objects']:
            
            if obj['name'].startswith(action_name['objectId']):
                action_name['objectId'] = obj['objectId']
                cnt -= 1
            elif 'receptableObjectId' in action_name and obj['name'].startswith(action_name['receptableObjectId']):
                action_name['receptableObjectId'] = obj['objectId']
                cnt -= 1
        if cnt ==0:
            env.step(action_name)        
            last_reward, done = env.get_transition_reward()
        else:
            last_reward = -0.2
            
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
        
    return sum(rewards_per_agent)

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
    # print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    # setup task for reward
    env.set_task(traj_data, args, reward_type=reward_type)


def train(worker_num,  shared_model, optimizer, lock, iter=None, split='train', gpu_id=-1):
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
    
        
    model = A3CLSTMNSingleAgent(num_inputs_per_agent=3, num_outputs=len(actions), state_repr_length=512, vocab=vocab)
    
    
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
                        
                        expert_reward = 0
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
                                r, _ = env.get_transition_reward()
                                expert_reward += r
                            else:
                                # print('object_poses', object_poses)
                                # print("object toggles", object_toggles)
                                # sub_task_object_poses = []
                                # for item in object_poses:
                                #     for it in env.last_event.metadata["objects"]:
                                #         if it["name"]  == item['objectName']:
                                #             entry = {}
                                #             entry['objectName'] = it["name"]
                                #             entry["rotation"] = it["rotation"]
                                #             entry["position"] = it["position"]
                                            
                                #             if it["isDirty"]:
                                #                 entry["ditry"] = True
                                #             else:
                                #                 entry["ditry"] = False
                                #             if it["isFilledWithLiquid"]:
                                #                 entry["filled"] = True
                                #             else:
                                #                 entry["filled"] = False
                                                
                                #             sub_task_object_poses.append(entry) 
                                #             print(it)
                              
                                # print("sub_task_object_poses", sub_task_object_poses)
                                
                                ## this will be our model
                                if lock and iter:
                                    with lock:
                                        iter.value += 1
                                reward = train_RL(env, model, shared_model, optimizer, lang_goal, actions, iter, gpu_id=gpu_id, mode=split)
                                if summary_writer:
                                    summary_writer.add_scalar('reward/' + split, reward, iter.value)
                                break
                                
                            t += 1

def construction_action_space():
    nouns = [ 
             'alarmclock', 'apple', 'armchair', 'baseballbat', 'basketball', 'bathtubbasin', 'bed', 'book', 'bottle', 'bowl', 'box', 'bread', 
              'butterknife', 'cabinet', 'candle', 'cart', 'cd', 'cellphone', 'cloth', 'coffeemachine', 'coffeetable', 'countertop', 'creditcard', 
              'cup', 'desk', 'desklamp', 'diningtable', 'dishsponge', 'drawer', 'dresser', 'egg', 'floorlamp', 'fork', 'fridge', 'garbagecan', 
              'handtowel', 'handtowelholder', 'kettle', 'keychain', 'knife', 'ladle', 'laptop', 'lettuce', 'microwave', 'mug', 'newspaper', 'ottoman', 
              'pan', 'pen', 'pencil', 'peppershaker', 'pillow', 'plate', 'plunger', 'pot', 'potato', 'remotecontrol', 'safe', 'saltshaker', 'shelf', 
              'sidetable', 'sinkbasin', 'soapbar', 'sofa', 'spatula', 'spoon', 'statue', 'stoveburner', 'tennisracket', 'tissuebox', 'toilet', 
              'toiletpaper', 'toiletpaperhanger', 'tomato', 'vase', 'watch', 'wateringcan'
        ]
    
    openable_nouns = [
        "Blinds",
        "Book",
        "Box",
        "Cabinet",
        "Drawer",
        "Fridge",
        "Kettle",
        "Laptop",
        "LaundryHamperLid",
        "Microwave",
        "Safe",
        "ShowerCurtain",
        "ShowerDoor",
    ]
    
    pickupable_nouns = [
        "AlarmClock",
        "Apple",
        "AppleSliced"
        "BaseballBat",
        "BasketBall",
        "Book",
        "Boots",
        "Bottle",
        "Bowl",
        "Box",
        "Bread",
        "BreadSliced",
        "ButterKnife",
        "Candle",
        "CD",
        "CellPhone",
        "Cloth",
        "CreditCard",
        "Cup",
        "DishSponge",
        "Egg",
        "EggCracked",
        "Fork",
        "HandTowel",
        "Kettle",
        "KeyChain",
        "Knife",
        "Ladle",
        "Laptop",
        "Lettuce",
        "LettuceSliced",
        "Mug",
        "Newspaper",
        "Pan",
        "PaperTowel",
        "Pen",
        "Pencil",
        "PepperShaker",
        "Pillow",
        "Plate",
        "Plunger",
        "Pot",
        "Potato",
        "PotatoSliced",
        "RemoteControl",
        "SaltShaker",
        "ScrubBrush",
        "SoapBar",
        "SoapBottle",
        "Spatula",
        "Spoon",
        "SprayBottle",
        "Statue",
        "TeddyBear",
        "TennisRacket",
        "TissueBox",
        "ToiletPaper",
        "Tomato",
        "TomatoSliced",
        "Towel",
        "Vase",
        "Watch",
        "WateringCan",
        "WineBottle",
    ]
    receptable_nouns = [
        "ArmChair",
        "Bathtub",
        "BathtubBasin",
        "Bed",
        "Bowl",
        "Box",
        "Cabinet",
        "Cart",
        "CoffeeMachine",
        "CoffeeTable",
        "CounterTop",
        "Cup",
        "Desk",
        "DiningTable",
        "Drawer",
        "Dresser",
        "Fridge",
        "GarbageCan",
        "HandTowelHolder",
        "LaundryHamper",
        "Microwave",
        "Mug",
        "Ottoman",
        "Pan",
        "Plate",
        "Pot",
        "Safe",
        "Shelf",
        "SideTable",
        "Sink",
        "SinkBasin",
        "Sofa",
        "StoveBurner",
        "Toaster",
        "Toilet"
        "ToiletPaperHanger",
        "TowelHolder",
        "TVStand"
    ]
    

    toggleable_nouns = [
        "Candle",
        "CellPhone",
        "CoffeeMachine",
        "DeskLamp",
        "Faucet",
        "FloorLamp",
        "Laptop",
        "LightSwitch",
        "Microwave",
        "ShowerHead",
        "StoveBurner",
        "StoveKnob",
        "Television",
        "Toaster",
        
    ]
    
    sliceable_nouns = [
        "Apple",
        "Bread",
        "Egg",
        "Lettuce",
        "Potato",
        "Tomato"
    ]
    verbs = [
        'MoveAhead',
        'RotateRight',
        'RotateLeft',
        'LookUp',
        'LookDown',
        'PickupObject',
        'PutObject',
        'OpenObject',
        'CloseObject',
        'ToggleObjectOn',
        'ToggleObjectOff',
        'SliceObject',
    ]
    nouns = [item.capitalize() for item in nouns]
    def compute_action(v, n1 = None, n2 = None):
        action = {}
        action['action']=v
        if n1:
            action['objectId'] = n1
        if n1 and n2:
            action['receptableId'] = n2
        return action
    
    action_space = []
    for i in range(5):
        action_space.append(compute_action(verbs[i]))
    
    for i in range(5,12):
        if verbs[i] == 'PutObject':
            for item1 in pickupable_nouns:
                for item2 in receptable_nouns: 
                    action_space.append(compute_action(verbs[i], item1, item2))
        elif verbs[i] == "PickupObject":
            for item in pickupable_nouns:
                action_space.append(compute_action(verbs[i], item))
        elif verbs[i] == 'OpenObject' or verbs[i] == 'CloseObject':
            for item in openable_nouns:
                action_space.append(compute_action(verbs[i], item))
                
        elif verbs[i] == 'ToggleObjectOn' or verbs[i] == 'ToggleObjectOff':
            for item in toggleable_nouns:
                action_space.append(compute_action(verbs[i], item))
        elif verbs[i] == 'SliceObject':
            for item in sliceable_nouns:
                action_space.append(compute_action(verbs[i], item)) 
    
    return action_space
    
if __name__ == '__main__':
    import ctypes
    import time
    
    vocab = torch.load('pp.vocab')
    # with open('actions.pickle', 'rb') as fp:
    #     actions = pickle.load(fp)
    #     actions = [eval(action) for action in actions]
    actions = construction_action_space()
    # print(actions)
    print(len(actions))
    exit()
    shared_model = A3CLSTMNSingleAgent(num_inputs_per_agent=3, num_outputs=len(actions), state_repr_length=512, vocab=vocab)
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
    valid_total_ep_train = mp.Value( ctypes.c_int32, 0 )
    valid_total_ep_valid_seen = mp.Value( ctypes.c_int32, 0 )
    valid_total_ep_valid_unseen = mp.Value( ctypes.c_int32, 0 )
    lock = mp.Lock()
    # train(0,shared_model,optimizer, gpu_id=0)
    # train(0, shared_model, optimizer, lock, valid_total_ep_train,  'train', 0 , summary_writer)
    for i in range(1):
        p = mp.Process(
                target=train,
                args=(
                    i,
                    shared_model,
                    optimizer,
                    lock,
                    valid_total_ep_train,
                    'train',
                    0,
                ),
            )

        p.start()
        processes.append(p)
        time.sleep(0.2)

   
    # p = mp.Process(
    #         target=train,
    #         args=(
    #             i,
    #             shared_model,
    #             optimizer,
    #             lock,
    #             valid_total_ep_valid_seen,
    #             'valid_seen',
    #             0,
               
    #         ),
    #     )

    # p.start()
    # processes.append(p)
    # time.sleep(0.2)
    
    
    # p = mp.Process(
    #     target=train,
    #     args=(
    #         i,
    #         shared_model,
    #         optimizer,
    #         lock,
    #         valid_total_ep_valid_unseen,
    #         'valid_unseen',
    #         0,
            
    #     ),
    # )

    # p.start()
    # processes.append(p)
    # time.sleep(0.2)
    for process in processes:
        process.join()
