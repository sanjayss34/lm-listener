import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_scheduler

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
from argparse import Namespace
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir = './dataset/HumanML3D'
if args.dataname == 'kit':
    args.vq_dir = './dataset/KIT-ML'
elif args.dataname.split('_')[0] == 'face':
    args.vq_dir = './dataset/'+args.dataname.split('_')[1]
args.vq_dir = os.path.join(args.vq_dir, args.vq_name)

os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
with open(os.path.join(args.out_dir, "config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4, sort_keys=True)

##### ---- Dataloader ---- #####
eval_split = "val"
if args.test_eval:
    eval_split = "test"
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t, max_motion_length=args.max_motion_length, split="train", delay_start_frames=args.delay_start_frames, fps=args.fps[0], min_length=args.train_min_length)
val_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t, max_motion_length=args.max_motion_length, split=eval_split, delay_start_frames=args.delay_start_frames, fps=args.fps[0], min_length=args.val_min_length)


##### ---- Network ---- #####
if args.gpt2 is None:
    if "openai/clip" in args.text_model_name:
        text_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)
    else:
        text_model = AutoModel.from_pretrained(args.text_model_name).cuda()
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    for p in text_model.parameters():
        p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

args.extra_input_dim={}
if args.gpt2 is not None:
    trans_encoder = trans.GPT2MotionTransformer(num_vq=args.nb_code, num_input_vq=(0 if args.speaker_vq_path is None else speaker_vq_args.nb_code), model_name=args.gpt2, top_p=args.top_p, extra_input_dim=args.extra_input_dim, freeze_lm=args.freeze_lm, output_layers=args.num_output_layers, not_pretrained=args.transformer_not_pretrained, gradient_checkpointing=args.gradient_checkpointing, predict_input_vq=args.speaker_vq_loss)
    text_model = trans_encoder
    text_tokenizer = None
    gpt2_config = AutoConfig.from_pretrained(args.gpt2)
else:   
    trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                    embed_dim=args.embed_dim_gpt, 
                                    clip_dim=args.clip_dim, 
                                    block_size=args.block_size, 
                                    num_layers=args.num_layers, 
                                    n_head=args.n_head_gpt, 
                                    drop_out_rate=args.drop_out_rate, 
                                    fc_rate=args.ff_rate,
                                    extra_dim=args.extra_input_dim,
                                    top_p=args.top_p)
    gpt2_config = None


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(utils_model.convert_vq_state_dict(ckpt['net']), strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
if args.fp16_half:
    trans_encoder = trans_encoder.half()
if args.manual_bf16:
    trans_encoder = trans_encoder.bfloat16()
trans_encoder.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
if args.linear_scheduler:
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=args.warm_up_iter, num_training_steps=args.total_iter//args.gradient_accumulation_steps)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
scaler = torch.cuda.amp.GradScaler()

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

speaker_vq_suffix = "_ofspeaker"
if args.total_iter > 0 or len(list(os.listdir(args.vq_dir))) == 0:
    ##### ---- get code ---- #####
    for batch in train_loader_token:
        pose, name = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
        target = net.encode(pose)
        target = target.cpu().numpy()
        os.system('mkdir -p '+pjoin(args.vq_dir, *name[0].split('/')[:-1]))
        np.save(pjoin(args.vq_dir, name[0] +'.npy'), target)

if args.total_iter > 0 or (not os.path.exists(args.vq_dir+"_"+eval_split)) or len(list(os.listdir(args.vq_dir+"_"+eval_split))):
    for batch in val_loader_token:
        pose, name = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.cuda().float()
        target = net.encode(pose)
        target = target.cpu().numpy()
        os.system('mkdir -p '+pjoin(args.vq_dir+'_'+eval_split, *name[0].split('/')[:-1]))
        np.save(pjoin(args.vq_dir+'_'+eval_split, name[0]+'.npy'), target)


train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, unit_length=2**args.down_t, split="train", max_motion_length=args.max_motion_length, 
                                           evaluation=False, gpt2_config=gpt2_config, no_text=args.no_text, max_tokens=args.max_tokens, no_before_text=args.no_before_text, max_time_before=args.max_time_before, 
                                           fps=args.fps[0], 
                                           fixed_text_token=args.fixed_text_token, fixed_text_token_not_space=args.fixed_text_token_not_space, fixed_text_token_not_punctuation=args.fixed_text_token_not_punctuation, unaligned_text=args.unaligned_text, remove_space_before_vq_tokens=args.remove_space_before_vq_tokens, random_text_token_order=args.random_text_token_order)
train_loader_iter = dataset_TM_train.cycle(train_loader)

val_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name+'_'+eval_split if not args.train_eval else args.vq_name, unit_length=2**args.down_t, split=eval_split if not args.train_eval else "train", max_motion_length=args.max_motion_length, 
                                         evaluation=True, gpt2_config=gpt2_config, no_text=args.no_text, max_tokens=args.max_tokens, no_before_text=args.no_before_text, max_time_before=args.max_time_before,
                                         fps=args.fps[0], 
                                         fixed_text_token=args.fixed_text_token, fixed_text_token_not_space=args.fixed_text_token_not_space, fixed_text_token_not_punctuation=args.fixed_text_token_not_punctuation, unaligned_text=args.unaligned_text, remove_space_before_vq_tokens=args.remove_space_before_vq_tokens, random_text_token_order=args.random_text_token_order)
        
##### ---- Training ---- #####
best_acc, best_loss, best_v_loss, best_windowed_v_loss, best_a_loss, best_e_loss, best_l2, best_iter, writer, logger = eval_trans.evaluation_transformer2(args, args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_acc=0, best_loss=float("inf"), best_v_loss=float("inf"), best_windowed_v_loss=float("inf"), best_a_loss=float("inf"), best_e_loss=float("inf"), best_l2=float("inf"), best_iter=0, text_model=text_model, text_tokenizer=text_tokenizer, max_motion_length=args.max_motion_length, draw=True, save=(args.total_iter > 0), savenpy=True, save_name=args.save_name, valence_window_size=args.valence_window_size, num_samples=args.num_samples)
optimizer.zero_grad()
prev_loss_total = float("inf")
curr_loss_total = 0.0
while nb_iter < args.total_iter:
    batch = next(train_loader_iter)
    if len(batch) == 4:
        before_text, during_text, m_tokens, m_tokens_len = batch
        input_text = (before_text, during_text)
    else:
        input_text, m_tokens, m_tokens_len = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens    # (bs, 26)
    target = target.cuda()
   
    with torch.cuda.amp.autocast(enabled=args.fp16):
        if not isinstance(input_text, tuple):
            input_text = (input_text,)
        text_feats = []
        if args.no_text:
            feat_clip_text = torch.zeros((bs, args.clip_dim)).float().to(m_tokens.device)
        elif args.gpt2 is None:
            for txt in input_text:
                if args.text_token_level and isinstance(txt[0], list):
                    char_indices = [[[] for _ in range(target.shape[1])] for _ in range(bs)]
                    full_texts = ["" for _ in range(bs)]
                    for j in range(bs):
                        for t in range(len(txt[j])):
                            if len(txt[j][t]) > 0:
                                for word in txt[j][t]:
                                    if len(full_texts[j]) > 0:
                                        full_texts[j] += " "
                                    char_indices[j][t].append(len(full_texts[j]))
                                    full_texts[j] += word
                    text_inputs = text_tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True).to(m_tokens.device)
                    with torch.no_grad():
                        if "openai/clip" in args.text_model_name:
                            feats_clip_text = text_model.encode_text(text_inputs.input_ids)
                        else:
                            feats_clip_text = text_model(**text_inputs).last_hidden_state
                    feats = torch.zeros_like((bs, target.shape[1], text_model.config.hidden_dim), dtype=torch.float32).to(m_tokens.device)
                    for j in range(bs):
                        for t in range(len(char_indices[j])):
                            if len(char_indices[j][t]) > 0:
                                feats[j,t,:] = feats_clip_text[j,[tok for c in char_indices[j][t] for tok in text_inputs[j].char_to_token(c)],:].mean(dim=0)
                    text_feats.append(feats)
                else:
                    text_inputs = text_tokenizer(txt, return_tensors='pt', padding=True, truncation=True).to(m_tokens.device)
                    with torch.no_grad():
                        text_inputs = clip.tokenize(txt)
                        feat_clip_text = text_model.encode_text(text_inputs)
                    text_feats.append(feat_clip_text)
            feat_clip_text = torch.cat(text_feats, dim=1)
            if args.manual_bf16:
                feat_clip_text = feat_clip_text.bfloat16()


        if args.gpt2 is not None:
            input_index = target
        else:
            input_index = target[:,:-1]

        if args.pkeep == -1:
            proba = np.random.rand(1)[0]
            mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                             device=input_index.device))
        else:
            mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                             device=input_index.device))
        mask = mask.round().to(dtype=torch.int64)
        if args.gpt2 is not None:
            r_indices = torch.where(
                (input_index >= text_model.text_vocab_size) & (input_index < text_model.text_vocab_size+args.nb_code),
                torch.randint_like(input_index, low=text_model.text_vocab_size, high=text_model.text_vocab_size+args.nb_code),
                input_index
            )
        else:
            r_indices = torch.randint_like(input_index, args.nb_code)
        a_indices = mask*input_index+(1-mask)*r_indices
        base_codebook_num = text_model.text_vocab_size+args.nb_code

        if args.gpt2 is not None:
            if (args.include_speaker and args.speaker_vq_path is None) or (args.include_audio and args.audio_vq_path is None):
                if args.fix_pkeep:
                    input_idx = a_indices
                else:
                    input_idx = input_index
                input_embeds = trans_encoder.gpt.transformer.wte(input_idx.clamp(min=0))
                if args.include_audio:
                    for i in range(input_idx.shape[0]):
                        num_audio_inputs = (input_idx[i] == -2).long().sum().item()
                        input_embeds[i,input_idx[i] == -2,:] = trans_encoder.extra_input_layers["aud"](audio_inputs[0][i,:num_audio_inputs,:]).to(input_embeds.dtype)
                    if args.audio_pkeep is not None:
                        audio_keep = (input_idx == -2) & (torch.rand_like(input_idx.float()) < args.audio_pkeep)
                        input_embeds = torch.where(
                            audio_keep.unsqueeze(-1).repeat(1, 1, input_embeds.shape[-1]),
                            input_embeds,
                            torch.zeros_like(input_embeds)
                        )
                if args.include_speaker: # search for -1 embeddings
                    for i in range(input_idx.shape[0]):
                        num_speaker_inputs = (input_idx[i] == -1).long().sum().item()
                        input_embeds[i,input_idx[i] == -1,:] = trans_encoder.extra_input_layers["mot"](speaker_inputs[0][i,:num_speaker_inputs,:]).to(input_embeds.dtype)
                    if args.speaker_pkeep is not None:
                        speaker_keep = (input_idx == -1) & (torch.rand_like(input_idx.float()) < args.speaker_pkeep)
                        input_embeds = torch.where(
                            speaker_keep.unsqueeze(-1).repeat(1, 1, input_embeds.shape[-1]),
                            input_embeds,
                            torch.zeros_like(input_embeds)
                        )
            else:
                input_embeds = trans_encoder.gpt.transformer.wte(a_indices)
            cls_pred = trans_encoder(input_ids=a_indices, input_embeds=input_embeds, attention_mask=m_tokens_len, predict_input_vq=args.speaker_vq_loss)

        cls_pred = cls_pred.contiguous()

        loss_cls = 0.0
        for i in range(bs):
            if args.gpt2 is not None:
                length_i = m_tokens_len[i].sum().item()
                if args.speaker_vq_loss:
                    mask_i = (target[i][1:length_i] >= text_model.text_vocab_size)
                else:
                    mask_i = (target[i][1:length_i] >= text_model.text_vocab_size) & (target[i][1:length_i] < text_model.text_vocab_size + args.nb_code)
                loss_cls += loss_ce(cls_pred[i][:length_i-1][mask_i], target[i][1:length_i][mask_i]-text_model.text_vocab_size) / bs
                probs = torch.softmax(cls_pred[i][:length_i-1], dim=-1)

            if args.if_maxtest:
                _, cls_pred_index = torch.max(probs, dim=-1)

            else:
                with torch.no_grad():
                    dist = Categorical(probs.float())
                    cls_pred_index = dist.sample()
            if args.gpt2 is not None:
                right_num += (cls_pred_index[mask_i] == target[i][1:length_i][mask_i]-text_model.text_vocab_size).sum().item()
                nb_sample_train += mask_i.long().sum().item()

        loss_cls = loss_cls / args.gradient_accumulation_steps
    curr_loss_total = curr_loss_total + loss_cls.item()
    if args.grad_scaling:
        scaler.scale(loss_cls).backward()
    else:
        loss_cls.backward()
    if ((nb_iter + 1) % args.gradient_accumulation_steps == 0) or (nb_iter + 1 == args.total_iter):
        if args.training_end_check_interval is not None and (nb_iter + 1) % (args.training_end_check_interval * args.gradient_accumulation_steps) == 0:
            if (prev_loss_total - curr_loss_total) / args.training_end_check_interval < args.train_loss_threshold:
                break
            prev_loss_total = curr_loss_total
            curr_loss_total = 0.0
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trans_encoder.parameters(), args.grad_clip)
        if args.grad_scaling:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    if args.gpt2 is None:
        nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter ==  0:
        best_acc, best_loss, best_v_loss, best_windowed_v_loss, best_a_loss, best_e_loss, best_l2, best_iter, writer, logger = eval_trans.evaluation_transformer2(args, args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_acc=best_acc, best_loss=best_loss, best_v_loss=best_v_loss, best_windowed_v_loss=best_windowed_v_loss, best_a_loss=best_a_loss, best_e_loss=best_e_loss, best_l2=best_l2, best_iter=best_iter, text_model=text_model, text_tokenizer=text_tokenizer, max_motion_length=args.max_motion_length, draw=True, save=True, savenpy=True, save_name=args.save_name, valence_window_size=args.valence_window_size, num_samples=args.num_samples)

    if nb_iter == args.total_iter: 
        msg_fiinal = f"Train. Iter {best_iter} : Acc. {best_acc:.5f}"
        logger.info(msg_final)
        break            
