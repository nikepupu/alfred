import numpy as np
from env.thor_env import ThorEnv
from tqdm import tqdm
import os
import json
import argparse


def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
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


if __name__ == '__main__':
    split = 'valid_seen'
    tasks = os.listdir('data/full_2.1.0/' + split)
    parser = argparse.ArgumentParser()

    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--num_threads', type=int, default=1)

    # eval params
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')


    args = parser.parse_args()
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
                    env = ThorEnv()
                    env.reset(scene_name)
                    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

                    # initialize to start position
                    env.step(dict(traj_data['scene']['init_action']))

                    # print goal instr
                    print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

                    # setup task for reward
                    env.set_task(traj_data, args, reward_type=reward_type)
            
                    setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

                    # expert demonstration to reach eval_idx-1
                    expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]

                    # subgoal info
                    subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
                    subgoal_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]

                    # print subgoal info
                    print("Evaluating: %s\nSubgoal %s (%d)\nInstr: %s" % (traj_data['root'], subgoal_action, eval_idx, subgoal_instr))
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