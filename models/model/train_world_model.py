
import os
import torch
import numpy as np
from torch.optim import optimizer
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from model.seq2seq_im_mask import Module as Base
from model.action_model import ContrastiveSWM, ModelFreePolicy
from model.action_model import a3c_loss
import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange
import warnings
from torch import optim
from typing import Tuple, Dict, Union, Optional, List, Any, Sequence

def compute_losses_and_backprop(
    loss_dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    retain_graph: bool = False,
    skip_backprop: bool = False,
    keep_gradient_and_skip_backprop: bool = False
) -> Dict[str, float]:
    model.zero_grad()
    full_loss = None
    last_losses = {}
    if keep_gradient_and_skip_backprop:
        skip_backprop = True
    for k, loss in loss_dict.items():
        loss = loss.squeeze()
        if keep_gradient_and_skip_backprop:
            last_losses["loss/" + k] = loss
        else:
            last_losses["loss/" + k] = loss.item()
        if full_loss is None:
            full_loss = loss
        elif (full_loss.is_cuda == loss.is_cuda) and (
            not full_loss.is_cuda or full_loss.get_device() == loss.get_device()
        ):
            full_loss += loss
        else:
            warnings.warn("Loss {} is on a different device!".format(k))
    
    if full_loss is not None:
        if not skip_backprop:
            full_loss.backward(retain_graph=retain_graph)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3, "inf")

           
            optimizer.step()
     


    return last_losses

class Module(Base):
    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)
        
        # encoder and self-attention
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        decoder = vnn.ConvFrameMaskDecoderProgressMonitor if self.subgoal_monitoring else vnn.ConvFrameMaskDecoder
        self.world_model = ContrastiveSWM(
            embedding_dim=2,
            hidden_dim= 512,
            action_dim=800,
            input_dims= torch.Size((512,7,7)),
            num_objects=5,
            sigma=0.5,
            hinge=1,
            ignore_action=False,
            copy_action=False,
            encoder='large'
        )
        self.a3c_model = ModelFreePolicy(800)
        
        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)
        

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()
        
        # a3c model parameters
        self.a3c_gamma = 0.99
        self.a3c_tau = 1.0
        self.a3c_beta = 1e-2
        self.huber_delta = None
        self.gpu_id = 0
        
    # def act(self):
        
        
    def run_dyna(self, feat, optim):
       
        cont_lang, enc_lang = self.encode_lang(feat)
        frames = self.vis_dropout(feat['frames'])
        # print(feat.keys())
        # print(feat['action_low'])
        # print(feat['action_low'].shape)
        # print("frames ", frames.shape)
        batch_size = frames.shape[0]
        len_dataset = frames.shape[1]
        # print("cont_lang", cont_lang.shape)
        # print("enc_lang", enc_lang.shape)
        # print("action", feat['action_low'].shape)
        # exit()
        state_loss = 0
        
        reward_loss = 0
        rewards = []
        for i in range(len_dataset-1):
            for batch in range(batch_size):
                
                
                state_0 = cont_lang[batch].unsqueeze(0), torch.zeros_like(cont_lang[batch].unsqueeze(0))
                state_t = state_0
                obs = frames[batch, i, :].unsqueeze(0)
                with torch.no_grad():
                    image_state = self.world_model.obj_extractor(obs)
                    latent_state = self.world_model.obj_encoder(image_state)
                    
                hidden1 = torch.zeros(1, 1, 512).cuda()
                hidden2 = torch.zeros(1, 1, 512).cuda()
                self.hidden = tuple((hidden1, hidden2))
                self.entropy_per_agent = []
                self.values_per_agent = []
                self.rewards_per_agent = []
                self.log_prob_of_actions = []
                
                for t in range(1):
                    eval_result = self.a3c_model(latent_state, self.hidden, None)
                    self.hidden = eval_result.get("hidden")
                    

                    # Either marginal or central agent
                    logit_per_agent = eval_result["actor"][0].unsqueeze(0)
                        
                    probs_per_agent =  F.softmax(logit_per_agent, dim=1) 

                    log_probs_per_agent = F.log_softmax(logit_per_agent, dim=1) 
                    
                    eval_result["log_probs_per_agent"] = log_probs_per_agent

                    self.hidden = eval_result.get("hidden")
                
                    self.entropy_per_agent.append(
                                        -(log_probs_per_agent * probs_per_agent).sum().unsqueeze(0)
                                )
                    
                    action = probs_per_agent.multinomial(num_samples=1).item()

                    log_prob_of_action_per_agent = log_probs_per_agent.view(-1)[action]
                    self.values_per_agent.append(eval_result.get("critic"))
                    
                    with torch.no_grad():
                        
                        reward, state_t = self.world_model.predict_reward(None, torch.LongTensor([action]).cuda(), enc_lang[batch].unsqueeze(0),  state_t, latent_state)
                        latent_state = latent_state + self.world_model.transition_model(latent_state, torch.LongTensor([action]).cuda())
                        
                    self.rewards_per_agent.append(reward)
                    self.log_prob_of_actions.append(log_prob_of_action_per_agent)
                    
                eval_result = self.a3c_model(latent_state, self.hidden, None)
                future_reward_est = eval_result["critic"].item()
                
                a3c_losses = a3c_loss(
                    values= self.values_per_agent,
                    rewards= self.rewards_per_agent,
                    log_prob_of_actions= self.log_prob_of_actions,
                    entropies= self.entropy_per_agent,
                    future_reward_est=future_reward_est,
                    gamma=self.a3c_gamma,
                    tau=self.a3c_tau,
                    beta=self.a3c_beta,
                    gpu_id=self.gpu_id,
                    huber_delta=self.huber_delta,
                )
                rewards.append(self.rewards_per_agent)
                compute_losses_and_backprop(
                    loss_dict =  a3c_losses,
                    model = self.a3c_model,
                    optimizer= optim)
                
        return rewards
        
    def run_train(self, splits, args=None, optimizer=None):
        '''
        training loop
        '''

        # args
        args = args or self.args
        # splits
        train = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:16]
            valid_seen = valid_seen[:16]
            valid_unseen = valid_unseen[:16]

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # display dout
        print("Saving to: %s" % self.args.dout)
        best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0
        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            # p_train = {}
            total_train_loss = list()
            random.shuffle(train) # shuffle every epoch
            
            for batch, feat in self.iterate(train, args.batch):
                out = self.forward(feat)
                rewards = self.run_dyna(feat, optimizer)
                # preds = self.extract_preds(out, batch, feat)
                # p_train.update(preds)
                # loss = self.compute_loss(out, batch, feat)
                loss = {}
                loss['world_model_loss'] = feat['state_loss']
                loss['reward_loss'] = feat['reward_loss']
                loss['fake_reward'] = sum([sum(r) for r in rewards])
                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                # optimizer backward pass
                optimizer.zero_grad()
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += self.args.batch
                
               
            continue

            ## compute metrics for train (too memory heavy!)
            # m_train = {k: sum(v) / len(v) for k, v in m_train.items()}
            # m_train.update(self.compute_metric(p_train, train))
            # m_train['total_loss'] = sum(total_train_loss) / len(total_train_loss)
            # self.summary_writer.add_scalar('train/total_loss', m_train['total_loss'], train_iter)

            # compute metrics for valid_seen
            p_valid_seen, valid_seen_iter, total_valid_seen_loss, m_valid_seen = self.run_pred(valid_seen, args=args, name='valid_seen', iter=valid_seen_iter)
            m_valid_seen.update(self.compute_metric(p_valid_seen, valid_seen))
            m_valid_seen['total_loss'] = float(total_valid_seen_loss)
            self.summary_writer.add_scalar('valid_seen/total_loss', m_valid_seen['total_loss'], valid_seen_iter)

            # compute metrics for valid_unseen
            p_valid_unseen, valid_unseen_iter, total_valid_unseen_loss, m_valid_unseen = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=valid_unseen_iter)
            m_valid_unseen.update(self.compute_metric(p_valid_unseen, valid_unseen))
            m_valid_unseen['total_loss'] = float(total_valid_unseen_loss)
            self.summary_writer.add_scalar('valid_unseen/total_loss', m_valid_unseen['total_loss'], valid_unseen_iter)

            stats = {'epoch': epoch,
                     'valid_seen': m_valid_seen,
                     'valid_unseen': m_valid_unseen}

            # new best valid_seen loss
            if total_valid_seen_loss < best_loss['valid_seen']:
                print('\nFound new best valid_seen!! Saving...')
                fsave = os.path.join(args.dout, 'best_seen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_seen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout, 'valid_seen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_seen, valid_seen), f, indent=2)
                best_loss['valid_seen'] = total_valid_seen_loss

            # new best valid_unseen loss
            if total_valid_unseen_loss < best_loss['valid_unseen']:
                print('Found new best valid_unseen!! Saving...')
                fsave = os.path.join(args.dout, 'best_unseen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_unseen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout, 'valid_unseen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_unseen, valid_unseen), f, indent=2)

                best_loss['valid_unseen'] = total_valid_unseen_loss

            # save the latest checkpoint
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            ## debug action output json for train
            # fpred = os.path.join(args.dout, 'train.debug.preds.json')
            # with open(fpred, 'wt') as f:
            #     json.dump(self.make_debug(p_train, train), f, indent=2)

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
            pprint.pprint(stats)

    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, args.batch):
            out = self.forward(feat)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'loss_' + k
                m_dev[ln].append(v.item())
                self.summary_writer.add_scalar("%s/%s" % (name, ln), v.item(), dev_iter)
            sum_loss = sum(loss.values())
            self.summary_writer.add_scalar("%s/loss" % (name), sum_loss, dev_iter)
            total_loss.append(float(sum_loss.detach().cpu()))
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev
    def encode_goal(self, feat):
        '''
        encode goal
        '''
   
        emb_lang_goal_instr = feat['lang_goal']
        self.lang_dropout(emb_lang_goal_instr.data)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)
     
        return cont_lang_goal_instr, enc_lang_goal_instr
    
    def encode_instr(self, feat):
        '''
        encode goal+instr language
        '''
       
        emb_lang_goal_instr = feat['lang_instr']
        self.lang_dropout(emb_lang_goal_instr.data)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)
       
        return cont_lang_goal_instr, enc_lang_goal_instr
    
    def forward(self, feat, max_decode=300):
        # cont_lang, enc_lang = self.encode_lang(feat)
        # state_0 = cont_lang, torch.zeros_like(cont_lang)
        # frames = self.vis_dropout(feat['frames'])
        # res = self.dec(enc_lang, frames, max_decode=max_decode, gold=feat['action_low'], state_0=state_0)
        # feat.update(res)
        cont_lang, enc_lang = self.encode_lang(feat)
        frames = self.vis_dropout(feat['frames'])
        # print(feat.keys())
        # print(feat['action_low'])
        # print(feat['action_low'].shape)
        # print("frames ", frames.shape)
        batch_size = frames.shape[0]
        len_dataset = frames.shape[1]
        # print("cont_lang", cont_lang.shape)
        # print("enc_lang", enc_lang.shape)
        # print("action", feat['action_low'].shape)
        # exit()
        state_loss = 0
        state_0 = cont_lang, torch.zeros_like(cont_lang)
        state_t = state_0
        reward_loss = 0
        for i in range(len_dataset-1):
            # print("step ", i)
            # print("frame inside, ", frames[:,i,:].shape)
            # print("action inside ", feat['action_low'][:, i].shape)
            # print("next frame inside ",frames[:,i+1,:].shape )
            # self.world_model.predict_reward(frames[:,i,:],)
            before_progress = feat['subgoal_progress'][:,i]
            after_progress = feat['subgoal_progress'][:,i+1]
            reward, state_t = self.world_model.predict_reward(frames[:,i,:], feat['action_low'][:,i], enc_lang, state_t)
            state_loss += self.world_model.contrastive_loss(frames[:,i,:], feat['action_low'][:,i], frames[:,i+1,:])
            
            # print(after_progress-before_progress)
            # print(reward.flatten())
            reward_loss += nn.MSELoss()(reward.flatten(), after_progress-before_progress)
        
        feat['reward_loss'] = reward_loss
        feat['state_loss'] = state_loss
        
        
        return feat
    
    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            # sigmoid preds to binary mask
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

            task_id_ann = self.get_task_and_ann_id(ex)
            pred[task_id_ann] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
            }

        return pred
    

    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].view(-1)
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0]*p_alow_mask.shape[1], *p_alow_mask.shape[2:])[valid_idxs]
        flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)
        alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask, flat_alow_mask)
        losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses
    
def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # subgoal completion supervision
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            lang_goal_instr = lang_goal + lang_instr
            feat['lang_goal_instr'].append(lang_goal_instr)

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                im = torch.load(os.path.join(root, self.feat_pt))

                num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action
                num_feat_frames = im.shape[0]

                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    feat['frames'].append(im)

                # Full Dataset (contains filler frames)
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

                # low-level action mask
                if load_mask:
                    feat['action_low_mask'].append([self.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])

                # low-level valid interact
                feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])


        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
        
        return feat
