import os
import sys

import clip
import numpy as np
import torch
from scipy import linalg
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModel, pipeline
from collections import defaultdict

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric

log_affect = False
if log_affect:
    from gdl.utils.other import get_path_to_assets
    from gdl_apps.EmotionRecognition.utils.io import load_model, test
face_expressions = {0:'neutral',
                    1:'joy',
                    2:'sadness',
                    3:'surprise',
                    4:'fear',
                    5:'disgust',
                    6:'anger',
                    7:'contempt', # no exist
                    8:'none'}
inv_face_expressions = {face_expressions[key]: key for key in face_expressions}

'''
NOTE: 
you will need to change emoca emotion recogition to not process from image)
add:

if 'image' not in batch:
    values = batch
else:
    values = self.deca.encode(batch, training=False)

to EmoDeca.py at the begining of the forward()
'''

def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)

@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_commit, best_iter, best_recons, best_perplexity, recons_loss_fn, draw = True, save = True, savegif=False, savenpy=False) : 
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []

    commit_loss = 0
    recons_loss = 0
    total_perplexity = 0
    for batch in val_loader:
        motion, m_length, name = batch

        motion = motion.cuda()
        bs, seq = motion.shape[0], motion.shape[1]

        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())

            pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            rloss = recons_loss_fn(pred_pose, motion[i:i+1, :m_length[i]])
            recons_loss += rloss
            commit_loss += loss_commit
            total_perplexity += perplexity
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            draw_org.append(pose)
            draw_pred.append(pred_denorm)
            
            if savenpy:
                path = os.path.join(out_dir, name[i])
                for j in range(len(path.split('/'))-1):
                    if not os.path.exists('/'.join(path.split('/')[:j+1])):
                        os.system('mkdir '+'/'.join(path.split('/')[:j+1]))
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose[:, :m_length[i]])
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_denorm)

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

        nb_sample += bs

    avg_commit = commit_loss / nb_sample
    avg_recons = recons_loss / nb_sample
    avg_perplexity = total_perplexity / nb_sample
    logger.info(f"Eval. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
    
    if draw:
        writer.add_scalar('./Val/Perplexity', avg_perplexity, nb_iter)
        writer.add_scalar('./Val/Commit', avg_commit, nb_iter)
        writer.add_scalar('./Val/Recons', avg_recons, nb_iter)
    
    if avg_perplexity < best_perplexity:
        msg = f"--> --> \t Perplexity Improved from {best_perplexity:.5f} to {avg_perplexity:.5f} !!!"
        logger.info(msg)
        best_perplexity = avg_perplexity
        if save:
            torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best.pth'))

    if avg_commit < best_commit:
        msg = f"--> --> \t Commit Improved from {best_commit:.5f} to {avg_commit:.5f} !!!"
        logger.info(msg)
        best_commit = avg_commit

    if avg_recons < best_recons:
        msg = f"--> --> \t Recons Improved from {best_recons:.5f} to {avg_recons:.5f} !!!"
        logger.info(msg)
        best_recons = avg_recons

    if save:
        torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))
        if nb_iter % 100000 == 0:
            torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_iter'+str(nb_iter)+'.pth'))

    net.train()
    return best_perplexity, best_iter, best_commit, best_recons, writer, logger

def loss_sentiment(pred_dict, gt_dict, affect_model, valence_window_size):
    with torch.no_grad():
        pred_affect = affect_model(pred_dict)
        gt_affect = affect_model(gt_dict)
    valence_loss = torch.nn.MSELoss()(pred_affect["valence"], gt_affect["valence"])
    windowed_pred_valence = pred_affect["valence"].unfold(dimension=0, size=min(valence_window_size, pred_affect["valence"].shape[0]), step=1)
    windowed_gt_valence = gt_affect["valence"].unfold(dimension=0, size=min(valence_window_size, gt_affect["valence"].shape[0]), step=1)
    windowed_valence_loss = torch.nn.MSELoss()(windowed_pred_valence, windowed_gt_valence)
    arousal_loss = torch.nn.MSELoss()(pred_affect["arousal"], gt_affect["arousal"])
    gt_expr =  torch.argmax(gt_affect["expr_classification"], dim=1)
    expr_loss = torch.nn.CrossEntropyLoss()(pred_affect["expr_classification"], gt_expr)
    return valence_loss, windowed_valence_loss, arousal_loss, expr_loss, pred_affect["expr_classification"]

@torch.no_grad()
def evaluation_transformer2(args, out_dir, val_loader, net, trans, logger, writer, nb_iter, best_acc, best_loss, best_v_loss, best_windowed_v_loss, best_a_loss, best_e_loss, best_l2, best_iter, text_model, text_tokenizer, max_motion_length, draw = True, save = True, savenpy = False, save_name=None, valence_window_size=None, num_samples=1):
    trans.eval()
    nb_sample = 0
    draw_org = []
    draw_pred = []

    # NOTE: added affect model here
    if log_affect:
        model_name = 'EMOCA-emorec'
        path_to_models = get_path_to_assets() /"EmotionRecognition"
        path_to_models = path_to_models / "face_reconstruction_based" # for 3dmm model
        affect_model = load_model(Path(path_to_models) / model_name)
        affect_model.cuda().eval()
    # done adding affect model

    correct = 0
    total = 0
    total_loss = 0
    total_l2 = 0
    total_v_loss = 0 # valence loss
    total_windowed_v_loss = 0 # windowed valence loss
    total_a_loss = 0 # affect loss
    total_e_loss = 0 # expression ce loss
    predictions = defaultdict(list)
    loss_ce = torch.nn.CrossEntropyLoss()
    first_batch = True
    for batch in val_loader:
        if len(batch) == 9:
            before_text, during_text, m_tokens, m_tokens_len, gt_exp, gt_pose, gt_shape, gt_detail, name = batch
            input_text = (before_text, during_text)
        else:
            input_text, m_tokens, m_tokens_len, gt_exp, gt_pose, gt_shape, gt_detail, name = batch

        bs = m_tokens.shape[0]
        m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
        text_emb = None
        with torch.cuda.amp.autocast(enabled=args.fp16):
            if not isinstance(input_text, tuple):
                input_text = (input_text,)
            if args.no_text:
                text_emb = torch.zeros((bs, args.clip_dim)).float().to(m_tokens.device)
            elif args.gpt2 is None:
                text_feats = []
                for txt_index, txt in enumerate(input_text):
                    if args.text_token_level and isinstance(txt[0], list):
                        char_indices = [[[] for _ in range(m_tokens.shape[1])] for _ in range(bs)]
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
                        feats = torch.zeros_like((bs, m_tokens.shape[1], text_model.config.hidden_dim), dtype=torch.float32).to(m_tokens.device)
                        for j in range(bs):
                            for t in range(len(char_indices[j])):
                                if len(char_indices[j][t]) > 0:
                                    feats[j,t,:] = feats_clip_text[j,[tok for c in char_indices[j][t] for tok in text_inputs[j].char_to_token(c)],:].mean(dim=0)
                        text_feats.append(feats)
                    else:
                        text_inputs = text_tokenizer(txt, return_tensors='pt', padding=True, truncation=True).to(m_tokens.device)
                        with torch.no_grad():
                            if "openai/clip" in args.text_model_name:
                                i = 0
                                while i < len(txt):
                                    try:
                                        text_inputs = clip.tokenize(txt[i:i+1])
                                        i += 1
                                    except:
                                        if txt_index == 0:
                                            txt[i] = txt[i][1:]
                                        else:
                                            txt[i] = txt[i][:-1]
                                feat_clip_text = text_model.encode_text(text_inputs)
                            else:
                                feat_clip_text = text_model(**text_inputs).pooler_output
                        text_feats.append(feat_clip_text)
                text_emb = torch.cat(text_feats, dim=1)

            if args.manual_bf16 and text_emb is not None:
                text_emb = text_emb.bfloat16()
            target = m_tokens[:,:-1]

            if args.gpt2 is not None:
                input_embeds = trans.gpt.transformer.wte(m_tokens)
                teacher_forced_pred = trans(input_ids=m_tokens, input_embeds=input_embeds, attention_mask=m_tokens_len)
            else:
                teacher_forced_pred = trans(target, text_emb)
        for i in range(bs):
            if first_batch and i < 5 and args.print_val_pred:
                if args.gpt2 is not None:
                    length_i = m_tokens_len[i].sum().item()
                    m_tokens_mask = (m_tokens[i,1:length_i] >= text_model.text_vocab_size) & (m_tokens[i,1:length_i] < text_model.text_vocab_size+args.nb_code)
                    # print('A', teacher_forced_pred[i,:length_i-1][m_tokens_mask].argmax(dim=-1))
                    # print('B', m_tokens[i,1:length_i][m_tokens_mask]-text_model.text_vocab_size)
                else:
                    # print('A', teacher_forced_pred[i,:m_tokens_len[i]+1].argmax(dim=-1))
                    # print('B', m_tokens[i,:m_tokens_len[i]+1])
                    pass
            if savenpy:
                for _ in range(num_samples):
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        if args.gpt2 is not None:
                            input_ids = torch.where(
                                (m_tokens >= text_model.text_vocab_size) & (m_tokens < text_model.text_vocab_size+args.nb_code),
                                torch.ones_like(m_tokens)*text_model.text_vocab_size,
                                m_tokens
                            )
                            index_motion = trans.sample(input_ids=input_ids[i:i+1,:], attention_mask=m_tokens_len[i:i+1,:])
                        else:
                            index_motion = trans.sample(text_emb[i:i+1,:], False)
                    if index_motion.numel() > 0:
                        pred = net.forward_decoder(index_motion)
                        pred_denorm = val_loader.dataset.inv_transform(pred.detach().cpu().numpy())
                        predictions[name[i]].append({'pred': pred_denorm, 'gt': np.concatenate((gt_exp[i,:,:].cpu().numpy(), gt_pose[i,:,:].cpu().numpy(), gt_shape[i,:,:].cpu().numpy(), gt_detail[i,:,:].cpu().numpy()), axis=1), "before_text": before_text[i], "during_text": during_text[i], 'pred_code': index_motion.cpu().numpy()})
                
                        cut_point = pred_denorm.shape[1]
                        if log_affect:
                            pred_dict = {"expcode": torch.reshape(torch.from_numpy(pred_denorm[:,:,:50]).cuda(), (-1, 50)), 
                                        "posecode": torch.reshape(torch.from_numpy(pred_denorm[:,:,50:]).cuda(), (-1, 6)), 
                                        "shapecode": torch.reshape(gt_shape[i,:cut_point,:].cuda(), (-1, 100))}
                            gt_dict = {"expcode": torch.reshape(gt_exp[i,:cut_point,:].cuda(), (-1, 50)), 
                                    "posecode": torch.reshape(gt_pose[i,:cut_point,:].cuda(), (-1, 6)), 
                                    "shapecode": torch.reshape(gt_shape[i,:cut_point,:].cuda(), (-1, 100))}
                            valence_loss, windowed_valence_loss, arousal_loss, expr_loss, pred_expr = loss_sentiment(pred_dict, gt_dict, affect_model, valence_window_size)

                            total_v_loss += valence_loss
                            total_windowed_v_loss += windowed_valence_loss
                            total_a_loss += arousal_loss
                            total_e_loss += expr_loss

                        # done adding semantic stuff

                        # NOTE(EV: added L2 on recon stuff here)
                        total_l2 += torch.nn.MSELoss()(torch.from_numpy(pred_denorm).cuda(), torch.cat((gt_exp[[i],:cut_point,:], gt_pose[[i],:cut_point,:]),axis=-1).cuda())

            if args.gpt2 is not None:
                length_i = m_tokens_len[i].sum().item()
                m_tokens_mask = (m_tokens[i,1:length_i] >= text_model.text_vocab_size) & (m_tokens[i,1:length_i] < text_model.text_vocab_size+args.nb_code)
                correct += (teacher_forced_pred[i,:length_i-1][m_tokens_mask,:].argmax(dim=-1) == m_tokens[i,1:length_i][m_tokens_mask]-text_model.text_vocab_size).long().sum()
                total_loss += loss_ce(teacher_forced_pred[i,:length_i-1,:][m_tokens_mask,:], m_tokens[i,1:length_i][m_tokens_mask]-text_model.text_vocab_size) / bs
                total += (m_tokens_mask).long().sum().item()
            else:
                correct += (teacher_forced_pred[i,:m_tokens_len[i].item()].argmax(dim=-1) == m_tokens[i,:m_tokens_len[i].item()]).long().sum()
                total_loss += loss_ce(teacher_forced_pred[i][:m_tokens_len[i] + 1], m_tokens[i][:m_tokens_len[i] + 1]) / bs
                total += m_tokens_len[i].item()
        first_batch = False
    acc = correct / total

    logger.info(f"Eval. Iter {nb_iter} : \t Accuracy. {acc:.5f} \t Loss. {total_loss:.5f} \t VLoss. {total_v_loss:.5f} \t ALoss {total_a_loss:.5f} \t ELoss {total_e_loss:.5f} \t L2 {total_l2:.5f}")

    if acc > best_acc:
        msg = f"--> --> \t Accuracy Improved from {best_acc:.5f} to {acc:.5f} !!!"
        logger.info(msg)
        best_acc = acc
    if total_loss < best_loss:
        msg = f"--> --> \t Loss Improved from {best_loss:.5f} to {total_loss:.5f} !!!"
        logger.info(msg)
        best_loss = total_loss
        best_iter = nb_iter
        if save:
            torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_best.pth'))
    if savenpy:
        for s in range(num_samples):
            for name in predictions:
                if save_name is not None:
                    save_dir = os.path.join(out_dir, save_name)
                else:
                    save_dir = out_dir
                if num_samples > 1:
                    save_dir += "_"+str(s)
                path = os.path.join(save_dir, name+'_pred.npy')
                for j in range(len(path.split('/'))-1):
                    if not os.path.exists('/'.join(path.split('/')[:j+1])):
                        os.system('mkdir '+'/'.join(path.split('/')[:j+1]))
                print(os.path.join(save_dir, name+'_pred.npy'))
                np.save(os.path.join(save_dir, name+'_pred.npy'), predictions[name][s]['pred'])
                np.save(os.path.join(save_dir, name+'_pred_code.npy'), predictions[name][s]['pred_code'])
                np.save(os.path.join(save_dir, name+'_gt.npy'), predictions[name][s]['gt'])
                with open(os.path.join(save_dir, name+'_text.json'), 'w') as fout:
                    json.dump({"before_text": predictions[name][s]['before_text'], "during_text": predictions[name][s]['during_text']}, fout)
    if total_l2 < best_l2:
        msg = f"--> --> \t L2 Loss Improved from {best_l2:.5f} to {total_l2:.5f} !!!"
        logger.info(msg)
        best_l2 = total_l2
        if save:
            torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_best_l2.pth'))
    
    if draw:
        writer.add_scalar('./Val/total_v_loss', total_v_loss, nb_iter)
        writer.add_scalar('./Val/total_windowed_v_loss', total_windowed_v_loss, nb_iter)
        writer.add_scalar('./Val/total_a_loss', total_a_loss, nb_iter)
        writer.add_scalar('./Val/total_e_loss', total_e_loss, nb_iter)
        writer.add_scalar('./Val/total_l2', total_l2, nb_iter)
        writer.add_scalar('./Val/acc', acc, nb_iter)
        writer.add_scalar('./Val/total_loss', total_loss.detach().float(), nb_iter)

    if save:
        torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))
        if nb_iter % 20000 == 0:
            torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_iter'+str(nb_iter)+'.pth'))

    trans.train()
    return best_acc, best_loss, best_v_loss, best_windowed_v_loss, best_a_loss, best_e_loss, best_l2, best_iter, writer, logger

@torch.no_grad()        
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in val_loader:
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], False)
                except:
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            
            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], True)
                except:
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
