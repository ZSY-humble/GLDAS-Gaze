import os

import time
import sys
sys.path.append('../common')

from common import utils, metrics
from torch.distributions import Categorical
import torch, json
from tqdm import tqdm
import numpy as np
from os.path import join
import torch.nn.functional as F
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.distributions import Categorical

def _sanitize(s):
    """Sanitize string for safe path usage."""
    return str(s).replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_")

def _to_numpy_image(img_tensor):
    """Convert [C,H,W] or [1,C,H,W] tensor to [H,W,3] numpy, normalized 0~1."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().float()
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax - vmin > 1e-6:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = img * 0.0
    img = img.permute(1, 2, 0).numpy()  # [H,W,3]
    return img
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os
PATH_COLOR = 'yellow'
NODE_COLOR = 'yellow'
TEXT_COLOR = '#ffe866'
EDGE_COLOR = 'yellow'
OUTPUT_PPI = 100
CIRCLE_RADIUS_PX = 64

LINE_WIDTH_BASE = 2
MARKER_SIZE_BASE = 4
FONT_SIZE_BASE = 8
FONT_SCALE = 2.5
LINE_ALPHA = 0.3

def save_step_overlays(
    img_tensor, step_maps, step_points_norm=None,
    out_dir="scanpath_vis", prefix="sample",
    arrow_color=PATH_COLOR, start_color=NODE_COLOR, current_color=NODE_COLOR,
    head_rel=0.02, lw=LINE_WIDTH_BASE, draw_numbers=True
):
    """Overlay image, heatmap, arrows, and fixation points."""
    os.makedirs(out_dir, exist_ok=True)

    def _to_numpy_image(img_tensor):
        if img_tensor.dim() == 4:
            img_t = img_tensor[0]
        else:
            img_t = img_tensor
        img = img_t.detach().cpu().float()
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        vmin, vmax = float(img.min()), float(img.max())
        img = (img - vmin) / (vmax - vmin + 1e-6)
        return img.permute(1, 2, 0).numpy()

    base = _to_numpy_image(img_tensor)
    H, W = base.shape[:2]
    head_size = max(H, W) * head_rel

    for t, m in enumerate(step_maps, 1):
        heat = m.detach().cpu().numpy()
        hmin, hmax = float(heat.min()), float(heat.max())
        heat = (heat - hmin) / (hmax - hmin + 1e-6)

        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.imshow(base)
        ax.imshow(heat, alpha=LINE_ALPHA, cmap='jet')

        if step_points_norm is not None and len(step_points_norm) >= t:
            pts = [(float(xn)*W, float(yn)*H) for (xn, yn) in step_points_norm[:t]]
            for k in range(len(pts)-1):
                x0, y0 = pts[k]
                x1, y1 = pts[k+1]
                arr = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle='-|>', mutation_scale=head_size,
                    linewidth=lw * 1.2, color=arrow_color, alpha=0.3,
                    shrinkA=0, shrinkB=0,
                    zorder=5,
                )
                ax.add_patch(arr)
            x0, y0 = pts[0]
            circ0 = mpatches.Circle((x0, y0), radius=max(W, H)*0.008,
                                    fill=False, edgecolor=start_color,
                                    linewidth=lw, zorder=6)
            ax.add_patch(circ0)
            xc, yc = pts[-1]
            circt = mpatches.Circle((xc, yc), radius=max(W, H)*0.009,
                                    fill=True, edgecolor="white",
                                    facecolor=current_color,
                                    linewidth=1.0, zorder=7)
            ax.add_patch(circt)
            if draw_numbers:
                for idx, (x, y) in enumerate(pts, 1):
                    ax.text(x+2, y-2, f"{idx}",
                            color=TEXT_COLOR,
                            fontsize=FONT_SIZE_BASE * FONT_SCALE,
                            ha='left', va='bottom', zorder=8)

        ax.set_axis_off()
        plt.tight_layout(pad=0)
        out_path = os.path.join(out_dir, f"{prefix}_step_{t:02d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



def get_IOR_mask_torch(norm_x_t, norm_y_t, h, w, r, device=None):
    """IOR mask in Torch; norm_x_t, norm_y_t: [B] in [0,1]; return [B, h*w] bool."""
    if device is None:
        device = norm_x_t.device
    bs = norm_x_t.numel()
    x = norm_x_t * w
    y = norm_y_t * h
    yy = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1)  # [1,h,1]
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)  # [1,1,w]
    dist2 = (xx - x.view(bs, 1, 1))**2 + (yy - y.view(bs, 1, 1))**2
    mask = (dist2 <= (r * r))  # [B,h,w] bool
    return mask.view(bs, -1)

def get_IOR_mask(norm_x, norm_y, h, w, r):
    """IOR mask from fixations; norm_x,y: (B,); return [B, h*w] bool."""
    bs = len(norm_x)
    x, y = norm_x * w, norm_y * h
    Y, X = np.ogrid[:h, :w]
    X = X.reshape(1, 1, w)
    Y = Y.reshape(1, h, 1)
    x = x.reshape(bs, 1, 1)
    y = y.reshape(bs, 1, 1)
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    return torch.from_numpy(mask.reshape(bs, -1))


def scanpath_decode(model, img, task_ids, pa, sample_action=False, center_initial=True):
    """Return trajs, nonstop_trajs, per_sample_step_maps, per_sample_stop_probs."""
    bs = img.size(0)
    device = img.device

    with torch.no_grad():
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = model.encode(img)

    per_sample_step_maps = [[] for _ in range(bs)]
    per_sample_stop_probs = [[] for _ in range(bs)]

    if center_initial:
        normalized_fixs = torch.zeros(bs, 1, 2, device=device).fill_(0.5)
        action_mask = get_IOR_mask_torch(
            torch.ones(bs, device=device) * 0.5,
            torch.ones(bs, device=device) * 0.5,
            pa.im_h, pa.im_w, pa.IOR_radius, device=device
        )  # [B, H*W] bool
    else:
        normalized_fixs = torch.zeros(bs, 0, 2, device=device)
        action_mask = torch.zeros(bs, pa.im_h * pa.im_w, dtype=torch.bool, device=device)

    stop_flags = []  # list of [B]

    for t in range(pa.max_traj_length):
        with torch.no_grad():
            if t == 0 and not center_initial:
                ys = ys_high = torch.zeros(bs, 1, dtype=torch.long, device=device)
                padding = torch.ones(bs, 1, dtype=torch.bool, device=device)
            else:
                ys, ys_high = utils.transform_fixations(
    normalized_fixs.cpu(), None, pa, False, return_highres=True)
                ys, ys_high = ys.to(device), ys_high.to(device)
                padding = None

            out = model.decode_and_predict(
                dorsal_embs.clone(), dorsal_pos, dorsal_mask, high_res_featmaps,
                ys, padding, ys_high, task_ids
            )

            prob_2d = out['pred_fixation_map']  # [B,H,W] or [B,H*W]
            if prob_2d.dim() == 2:
                prob_2d = prob_2d.view(bs, pa.im_h, pa.im_w)

            stop = out['pred_termination'].view(-1)  # [B]

            for b in range(bs):
                per_sample_step_maps[b].append(prob_2d[b].detach().cpu())
                per_sample_stop_probs[b].append(float(stop[b].detach().cpu()))

            prob = prob_2d.view(bs, -1)  # [B, H*W]
            stop_flags.append(stop)

            if pa.enforce_IOR:
                prob[action_mask] = 0.0

        if sample_action:
            row_sum = prob.sum(dim=1, keepdim=True)  # [B,1]
            zero_rows = (row_sum <= 1e-12).squeeze(1)  # [B]
            if zero_rows.any():
                if pa.enforce_IOR:
                    mask_valid = (~action_mask).float()
                    mask_valid = mask_valid / (mask_valid.sum(dim=1, keepdim=True) + 1e-12)
                    prob[zero_rows] = mask_valid[zero_rows]
                else:
                    prob[zero_rows] = 1.0 / prob.size(1)
            probs_norm = prob / (prob.sum(dim=1, keepdim=True) + 1e-12)
            next_word = Categorical(probs=probs_norm).sample()
        else:
            _, next_word = torch.max(prob, dim=1)

        next_word = next_word.detach().cpu()
        norm_fy = (next_word // pa.im_w) / float(pa.im_h)
        norm_fx = (next_word % pa.im_w) / float(pa.im_w)

        normalized_fixs = torch.cat(
            [normalized_fixs,
             torch.stack([norm_fx, norm_fy], dim=1).to(device).unsqueeze(1)],
            dim=1
        )

        new_mask = get_IOR_mask_torch(
            norm_fx.to(device=device, dtype=torch.float32),
            norm_fy.to(device=device, dtype=torch.float32),
            pa.im_h, pa.im_w, pa.IOR_radius, device=device
        )
        action_mask = torch.logical_or(action_mask, new_mask)

    stop_flags = torch.stack(stop_flags, dim=1)

    trajs, nonstop_trajs = [], []
    for b in range(bs):
        is_terminal = (stop_flags[b] > 0.5)
        if is_terminal.any():
            ind = int(is_terminal.int().argmax().item()) + 1
        else:
            ind = normalized_fixs.size(1)
        trajs.append(normalized_fixs[b, :ind])        # [T_b,2]
        nonstop_trajs.append(normalized_fixs[b])      # [T_max,2]

    return trajs, nonstop_trajs, per_sample_step_maps, per_sample_stop_probs


def actions2scanpaths(norm_fixs, patch_num, im_h, im_w):
    # convert actions to scanpaths
    scanpaths = []
    for traj in norm_fixs:
        task_name, img_name, condition, fixs = traj
        fixs = fixs.cpu().numpy()
        scanpaths.append({
            'X': fixs[:, 0] * im_w,
            'Y': fixs[:, 1] * im_h,
            'name': img_name,
            'task': task_name,
            'condition': condition
        })
    return scanpaths


def compute_conditional_saliency_metrics(pa, model, gazeloader, task_dep_prior_maps, device):
    n_samples, info_gain, nss, auc = 0, 0, 0, 0
    for batch in tqdm(gazeloader, desc='Computing saliency metrics'):
        img = batch['true_state'].to(device)
        task_ids = batch['task_id'].to(device)
        is_last = batch['is_last']
        non_term_mask = torch.logical_not(is_last)
        if torch.sum(non_term_mask) == 0:
            continue
        inp_seq, inp_seq_high = utils.transform_fixations(
            batch['normalized_fixations'], batch['is_padding'],
            pa, False, return_highres=True)
        inp_seq = inp_seq.to(device)
        inp_padding_mask = (inp_seq == pa.pad_idx)

        gt_next_fixs = (batch['next_normalized_fixations'][:, -1] * torch.tensor(
            [pa.im_w, pa.im_h])).to(torch.long)
        prior_maps = torch.stack(
            [task_dep_prior_maps[task] for task in batch['task_name']]).cpu()
        with torch.no_grad():
            logits = model(img, inp_seq, inp_padding_mask, inp_seq_high.to(device), task_ids)
            pred_fix_map = logits['pred_fixation_map']
            if len(pred_fix_map.size()) > 3:
                pred_fix_map = pred_fix_map[torch.arange(img.size(0)), task_ids]
            pred_fix_map = pred_fix_map.detach().cpu()
            # pred_fix_map = torch.nn.functional.interpolate(
            #     pred_fix_map.unsqueeze(1), size=(pa.im_h, pa.im_w), mode='bilinear').squeeze(1)

            probs = pred_fix_map
            # Normalize values to 0-1
            # probs -= probs.view(probs.size(0), 1, -1).min(dim=-1, keepdim=True)[0]
            probs /= probs.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        probs = probs[non_term_mask]
        prior_maps = prior_maps[non_term_mask]
        gt_next_fixs = gt_next_fixs[non_term_mask]
        info_gain += metrics.compute_info_gain(probs, gt_next_fixs, prior_maps)
        nss += metrics.compute_NSS(probs, gt_next_fixs)
        auc += metrics.compute_cAUC(probs, gt_next_fixs)
        n_samples += gt_next_fixs.size(0)

    info_gain /= n_samples
    nss /= n_samples
    auc /= n_samples

    return info_gain.item(), nss.item(), auc.item()

def sample_scanpaths(model, dataloader, pa, device, sample_action, center_initial=True):
    """Return scanpaths and nonstop_actions."""
    all_actions = []
    nonstop_actions = []

    for i in range(10):
        for batch in tqdm(dataloader, desc=f'Generate scanpaths [{i}/10]:'):
            img = batch['im_tensor'].to(device)
            task_ids = batch['task_id'].to(device)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            cond_batch = batch['condition']

            trajs, nonstop_trajs, per_sample_maps, per_sample_stop = scanpath_decode(
                model.module if isinstance(model, torch.nn.DataParallel) else model,
                img, task_ids, pa, sample_action, center_initial
            )

            out_root = "scanpath_step_overlays"
            os.makedirs(out_root, exist_ok=True)
            for b in range(img.size(0)):
                name = f"{_sanitize(cat_names_batch[b])}__{_sanitize(img_names_batch[b])}__{_sanitize(cond_batch[b])}"
                out_dir = os.path.join(out_root, name)
                os.makedirs(out_dir, exist_ok=True)

                T = trajs[b].size(0)
                step_points_norm = [(float(trajs[b][t,0].cpu()), float(trajs[b][t,1].cpu()))
                                    for t in range(T)]

                save_step_overlays(
                    img_tensor=img[b],
                    step_maps=per_sample_maps[b][:T],
                    step_points_norm=step_points_norm,
                    out_dir=out_dir,
                    prefix="overlay"
                )

            nonstop_actions.extend([
                (cat_names_batch[k], img_names_batch[k], cond_batch[k], nonstop_trajs[k])
                for k in range(img.size(0))
            ])
            all_actions.extend([
                (cat_names_batch[k], img_names_batch[k], cond_batch[k], trajs[k])
                for k in range(img.size(0))
            ])

        if not sample_action:
            break

    scanpaths = actions2scanpaths(all_actions, pa.patch_num, pa.im_h, pa.im_w)
    return scanpaths, nonstop_actions

def evaluate(model, device,
             valid_img_loader, gazeloader, pa,
             bbox_annos, human_cdf, fix_clusters,
             task_dep_prior_maps, semSS_strings, dataset_root,
             human_scanpath_test,
             sample_action=False, sample_stop=False,
             output_saliency_metrics=True, center_initial=True, log_dir=None):
    print("Eval on {} batches of images and {} batches of fixations".format(
        len(valid_img_loader), len(gazeloader)))

    model.eval()
    TAP = pa.TAP
    cut1, cut2, cut3 = 2, 4, 6

    print(f"Evaluating {TAP} with max steps to be {pa.max_traj_length} " +
          f"with initial center fixation = {center_initial} " +
          f"and enforce IOR = {pa.enforce_IOR} with radius {pa.IOR_radius}...")

    scanpaths, nonstop_actions = sample_scanpaths(
        model, valid_img_loader, pa, device, sample_action, center_initial)

    nonstop_scanpaths = actions2scanpaths(nonstop_actions, pa.patch_num, pa.im_h, pa.im_w)

    print('Computing metrics...')
    metrics_dict = {}

    if TAP == 'TP':
        if not sample_stop:
            utils.cutFixOnTarget(scanpaths, bbox_annos)

        mean_cdf, _ = utils.compute_search_cdf(
            scanpaths, bbox_annos, pa.max_traj_length)

        metrics_dict.update(dict(zip([f"TFP_top{i}" for i in range(
            1, len(mean_cdf))], mean_cdf[1:])))

        metrics_dict['prob_mismatch'] = np.sum(np.abs(human_cdf[:len(mean_cdf)] - mean_cdf))

    ss = metrics.get_seq_score(scanpaths, fix_clusters, pa.max_traj_length, False)
    fed= metrics.get_ed(scanpaths, fix_clusters, pa.max_traj_length, False)
    mm_scores = metrics.compute_mm(
        human_scanpath_test, scanpaths, pa.im_w, pa.im_h, tasks=None
    )
    if isinstance(mm_scores, (list, tuple, np.ndarray)):
        mm_scores = np.mean(mm_scores)
    metrics_dict.update({
        f"{TAP}_seq_score_max": ss,
        f"{TAP}_fed_max": fed,
        f"{TAP}_mm_scores": mm_scores,
    })

    if semSS_strings is not None:
        sss = metrics.get_semantic_seq_score(
            scanpaths, semSS_strings, pa.max_traj_length,
            f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', False)
        
        semfed=metrics.get_semantic_ed(
            scanpaths, semSS_strings, pa.max_traj_length,
            f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', False)
        
        metrics_dict.update({
            f"{TAP}_semantic_seq_score_max": sss,
            f"{TAP}_semantic_fed_score_max": semfed,
        })

    if output_saliency_metrics:
        ig, nss, auc = compute_conditional_saliency_metrics(
            pa, model, gazeloader, task_dep_prior_maps, device)
        metrics_dict.update({
            f"{TAP}_cIG": ig,
            f"{TAP}_cNSS": nss,
            f"{TAP}_cAUC": auc,
        })

    sp_len_diff = []
    for traj in scanpaths:
        gt_trajs = list(filter(
            lambda x: x['task'] == traj['task'] and x['name'] == traj['name'],
            human_scanpath_test))
        sp_len_diff.append(len(traj['X']) - np.array([len(t['X']) for t in gt_trajs]))

    sp_len_diff = np.abs(np.concatenate(sp_len_diff))
    metrics_dict[f'{TAP}_sp_len_err_mean'] = sp_len_diff.mean()
    metrics_dict[f'{TAP}_sp_len_err_std'] = sp_len_diff.std()
    metrics_dict[f'{TAP}_avg_sp_len'] = np.mean([len(x['X']) for x in scanpaths])

    if not sample_action:
        prefix = 'Greedy_'
        keys = list(metrics_dict.keys())
        for k in keys:
            metrics_dict[prefix + k] = metrics_dict.pop(k)

    if log_dir is not None:
        for sp in scanpaths:
            sp['X'] = sp['X'].tolist()
            sp['Y'] = sp['Y'].tolist()

        predictions_filename = generate_unique_filename(join(log_dir, f'predictions_{TAP}.json'))
        with open(predictions_filename, 'w') as f:
            json.dump(scanpaths, f, indent=4)

        metrics_filename = generate_unique_filename(join(log_dir, f'metrics_{TAP}.json'))
        with open(metrics_filename, 'w') as f:
            metrics_dict_converted = convert_numpy_to_python(metrics_dict)
            json.dump(metrics_dict_converted, f, indent=4)

    return metrics_dict, scanpaths


def convert_numpy_to_python(obj):
    """Convert numpy types in dict to Python natives."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj

def generate_unique_filename(base_path):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    base_name, ext = os.path.splitext(base_path)
    return f"{base_name}_{timestamp}{ext}"

