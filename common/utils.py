import numpy as np
import torch
import math
import torch.nn.functional as F
from copy import copy
from torch.distributions import Categorical
import scipy.ndimage as filters
import warnings
import os
import re, cv2
from shutil import copyfile

warnings.filterwarnings("ignore", category=UserWarning)


def get_foveal_weights(fixation_batch,
                       width,
                       height,
                       sigma=0.248,
                       p=7.5,
                       k=1.5,
                       alpha=1.25):
    """
    This function generate foveated image in batch on GPU

    fixation_batch: normalized fixation tensor of shape (batch_size,
      fix_num, 2)
    """
    assert fixation_batch.size(-1) == 2, 'Wrong input shape!'
    assert fixation_batch.max() <= 1, 'Fixation has to be normalized!'
    prNum = 5

    batch_size = fixation_batch.size(0)
    fix_num = fixation_batch.size(1)
    device = fixation_batch.device

    # Map fixations to coordinate space
    fixation_batch = fixation_batch * torch.tensor([width, height]).to(device)

    x = torch.arange(0, width, device=device, dtype=torch.float)
    y = torch.arange(0, height, device=device, dtype=torch.float)
    y2d, x2d = torch.meshgrid([y, x])
    h, w = x2d.size()

    x2d = x2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)
    y2d = y2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)

    # fixation patch index to fixation pixel coordinates
    xc = fixation_batch[:, :, 0]
    yc = fixation_batch[:, :, 1]

    xc2d = xc.view(batch_size, fix_num, 1, 1).expand_as(x2d)
    yc2d = yc.view(batch_size, fix_num, 1, 1).expand_as(y2d)

    theta = torch.sqrt((x2d - xc2d) ** 2 + (y2d - yc2d) ** 2) / p
    theta, _ = torch.min(theta, dim=1)
    R = alpha / (theta + alpha)

    Ts = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum - 1):
        Ts[:, i] = torch.exp(-((2 ** (i - 2)) * R / sigma) ** 2 * k)

    # omega
    omega = torch.zeros(prNum)
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k)
    omega[:-1] = torch.sqrt(math.log(2) / k) / (2 ** torch.arange(
        -2, prNum // 2, dtype=torch.float32, device=device)) * sigma
    omega[omega > 1] = 1

    # layer index
    layer_ind = torch.zeros_like(R, device=device)
    for i in range(1, prNum):
        ind = (R >= omega[i]) * (R <= omega[i - 1])
        layer_ind[ind] = i

    # Bs
    Bs = (0.5 - Ts[:, 1:]) / (Ts[:, :-1] - Ts[:, 1:])

    # M
    Ms = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum):
        ind = layer_ind == i
        if torch.sum(ind) > 0:
            if i == 0:
                Ms[:, i][ind] = 1
            else:
                Ms[:, i][ind] = 1 - Bs[:, i - 1][ind]

        ind = layer_ind - 1 == i
        if torch.sum(ind) > 0:
            Ms[:, i][ind] = Bs[:, i][ind]

    return Ms


def cutFixOnTarget(trajs, target_annos):
    task_names = np.unique([traj['task'] for traj in trajs])

    if 'condition' in trajs[0].keys():
        trajs = list(filter(lambda x: x['condition'] == 'present', trajs))

    if len(trajs) == 0:
        return

    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)

        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]

            # 计算从开始到第一次注视命中目标所需的步数
            traj_len = get_num_step2target(traj['X'], traj['Y'], bbox)

            # 将该轨迹的有效长度记录下来
            num_steps_task[i] = traj_len

            # 将该轨迹的X/Y坐标截断为前traj_len步，仅保留到达目标前的部分
            traj['X'] = traj['X'][:traj_len]
            traj['Y'] = traj['Y'][:traj_len]



def pos_to_action(center_x, center_y, patch_size, patch_num):
    """
    将一个坐标点(center_x, center_y)映射到 patch 网格中的一个 index（action）

    参数:
        center_x, center_y: 注视点的坐标（像素单位）
        patch_size: [patch_width, patch_height]，每个 patch 的宽和高
        patch_num: [num_cols, num_rows]，patch 的列数和行数（注意顺序！）

    返回:
        一个整数，表示该注视点位于哪个 patch 上，对应的 action index
        该索引按“行优先”的方式编号（从左到右，从上到下）
    """

    # 计算该注视点落在哪个 patch 上（x为列，y为行）
    x = center_x // patch_size[0]  # 列索引
    y = center_y // patch_size[1]  # 行索引

    # 将二维位置映射为一维索引（行优先编码：第 y 行第 x 列 -> y * 列数 + x）
    return int(patch_num[0] * y + x)



def action_to_pos(acts, patch_size, patch_num):
    """
    将动作编号（action index）转换为图像中的像素坐标（patch中心点）

    参数:
        acts: 整数或整数列表，表示动作编号（patch 索引）
        patch_size: [patch_width, patch_height]，每个 patch 的大小
        patch_num: [num_cols, num_rows]，patch 的列数和行数（注意顺序！）

    返回:
        pixel_x, pixel_y: 该 patch 的中心像素坐标（浮点数）
    """

    # 根据行优先编号方式：先计算该动作编号对应的 patch 的行号（y方向）
    patch_y = acts // patch_num[0]  # 第几行（从上到下）

    # 再计算列号（x方向）
    patch_x = acts % patch_num[0]   # 第几列（从左到右）

    # 计算 patch 中心点的 x 坐标：列号 * patch 宽度 + 半个 patch 宽度
    pixel_x = patch_x * patch_size[0] + patch_size[0] / 2

    # 计算 patch 中心点的 y 坐标：行号 * patch 高度 + 半个 patch 高度
    pixel_y = patch_y * patch_size[1] + patch_size[1] / 2

    return pixel_x, pixel_y



def select_action(obs,
                  policy,
                  sample_action,
                  action_mask=None,
                  softmask=False,
                  eps=1e-12,
                  has_stop=False):
    probs, values = policy(*obs)
    if sample_action:
        m = Categorical(probs)
        if action_mask is not None:
            # prevent sample previous actions by re-normalizing probs
            probs_new = probs.clone().detach()
            if softmask:
                probs_new = probs_new * action_mask
            else:
                if has_stop:
                    probs_new[:, :-1][action_mask] = eps
                else:
                    probs_new[action_mask] = eps

            probs_new /= probs_new.sum(dim=1).view(probs_new.size(0), 1)

            m_new = Categorical(probs_new)
            actions = m_new.sample()
        else:
            actions = m.sample()
        log_probs = m.log_prob(actions)
        return actions.view(-1), log_probs, values.view(-1), probs
    else:
        probs_new = probs.clone().detach()
        if has_stop:
            probs_new[:, :-1][action_mask.view(probs_new.size(0), -1)] = 0
        else:
            probs_new[action_mask.view(probs_new.size(0), -1)] = 0
        actions = torch.argmax(probs_new, dim=1)
        return actions.view(-1), None, None, None


def collect_trajs(env,
                  policy,
                  patch_num,
                  max_traj_length,
                  is_eval=False,
                  sample_action=True,
                  is_zero_shot=False):
    rewards = []
    obs_fov = env.observe()
    is_composite_state = isinstance(obs_fov, tuple)

    def pack_model_inputs(obs_fov, env):
        if is_composite_state:
            inputs = [*obs_fov, env.task_ids]
        else:
            inputs = [obs_fov, env.task_ids]
        if is_zero_shot:
            inputs.append(env.hr_feats)
        return inputs

    inputs = pack_model_inputs(obs_fov, env)
    act, log_prob, value, prob = select_action(inputs,
                                               policy,
                                               sample_action,
                                               action_mask=env.action_mask,
                                               has_stop=env.pa.has_stop)
    status = [env.status]
    values = [value]
    log_probs = [log_prob]
    SASPs = []

    i = 0
    if is_eval:
        actions = []
        while i < max_traj_length:
            new_obs_fov, curr_status = env.step(act)
            status.append(curr_status)
            actions.append(act)
            obs_fov = new_obs_fov
            inputs = pack_model_inputs(obs_fov, env)
            act, log_prob, value, prob_new = select_action(
                inputs,
                policy,
                sample_action,
                action_mask=env.action_mask,
                has_stop=env.pa.has_stop)
            i = i + 1

        status = torch.stack(status[1:])
        actions = torch.stack(actions)

        bs = len(env.img_names)
        trajs = []
        for i in range(bs):
            ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
            if status[:, i].sum() == 0:
                ind = status.size(0)
            trajs.append({'actions': actions[:ind, i]})

    else:
        IORs = []
        IORs.append(
            env.action_mask.to(dtype=torch.float).view(env.batch_size, 1,
                                                       patch_num[1], -1))
        while i < max_traj_length and env.status.min() < 1:
            new_obs_fov, curr_status = env.step(act)

            status.append(curr_status)
            SASPs.append((obs_fov, act, new_obs_fov))
            obs_fov = new_obs_fov

            IORs.append(
                env.action_mask.to(dtype=torch.float).view(
                    env.batch_size, 1, patch_num[1], -1))
            inputs = pack_model_inputs(obs_fov, env)
            act, log_prob, value, prob_new = select_action(
                inputs,
                policy,
                sample_action,
                action_mask=env.action_mask,
                has_stop=env.pa.has_stop)
            values.append(value)
            log_probs.append(log_prob)

            # place holder, reward assigned after collection is done
            rewards.append(torch.zeros(env.batch_size))

            i = i + 1

        if is_composite_state:
            num_state_comps = len(SASPs[0][0])
            S = [
                torch.stack([sasp[0][i] for sasp in SASPs])
                for i in range(num_state_comps)
            ]
        else:
            S = torch.stack([sasp[0] for sasp in SASPs])
        A = torch.stack([sasp[1] for sasp in SASPs])
        V = torch.stack(values)
        R = torch.stack(rewards)
        LogP = torch.stack(log_probs[:-1])
        status = torch.stack(status[1:])

        bs = len(env.img_names)
        trajs = []

        for i in range(bs):
            ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
            if status[:, i].sum() == 0:
                ind = status.size(0)
            trajs.append({
                'curr_states':
                    [s[:ind, i] for s in S] if is_composite_state else S[:ind, i],
                'actions':
                    A[:ind, i],
                'values':
                    V[:ind + 1, i],
                'log_probs':
                    LogP[:ind, i],
                'rewards':
                    R[:ind, i],
                'task_id':
                    env.task_ids[i].repeat(ind),
                'img_name': [env.img_names[i]] * ind,
                'length':
                    ind,
                'hr_feats':
                    torch.stack([env.hr_feats[i]] * ind) if is_zero_shot else None
            })

    return trajs


def compute_return_advantage(rewards, values, gamma, mtd='CRITIC', tau=0.96):
    device = rewards.device
    acc_reward = torch.zeros_like(rewards, dtype=torch.float, device=device)
    acc_reward[-1] = rewards[-1]
    for i in reversed(range(acc_reward.size(0) - 1)):
        acc_reward[i] = rewards[i] + gamma * acc_reward[i + 1]

    # compute advantages
    if mtd == 'MC':  # Monte-Carlo estimation
        advs = acc_reward - values[:-1]
    elif mtd == 'CRITIC':  # critic estimation
        advs = rewards + gamma * values[1:] - values[:-1]
    elif mtd == 'GAE':  # generalized advantage estimation
        delta = rewards + gamma * values[1:] - values[:-1]
        adv = torch.zeros_like(delta, dtype=torch.float, device=device)
        adv[-1] = delta[-1]
        for i in reversed(range(delta.size(0) - 1)):
            adv[i] = delta[i] + gamma * tau * adv[i + 1]
    else:
        raise NotImplementedError

    return acc_reward.squeeze(), advs.squeeze()


def process_trajs(trajs, gamma, mtd='CRITIC', tau=0.96):
    # compute discounted cummulative reward
    device = trajs[0]['log_probs'].device
    avg_return = 0
    for traj in trajs:

        acc_reward = torch.zeros_like(traj['rewards'],
                                      dtype=torch.float,
                                      device=device)
        acc_reward[-1] = traj['rewards'][-1]
        for i in reversed(range(acc_reward.size(0) - 1)):
            acc_reward[i] = traj['rewards'][i] + gamma * acc_reward[i + 1]

        traj['acc_rewards'] = acc_reward
        avg_return += acc_reward[0]

        values = traj['values']
        # compute advantages
        if mtd == 'MC':  # Monte-Carlo estimation
            traj['advantages'] = traj['acc_rewards'] - values[:-1]

        elif mtd == 'CRITIC':  # critic estimation
            traj['advantages'] = traj[
                                     'rewards'] + gamma * values[1:] - values[:-1]

        elif mtd == 'GAE':  # generalized advantage estimation
            delta = traj['rewards'] + gamma * values[1:] - values[:-1]
            adv = torch.zeros_like(delta, dtype=torch.float, device=device)
            adv[-1] = delta[-1]
            for i in reversed(range(delta.size(0) - 1)):
                adv[i] = delta[i] + gamma * tau * adv[i + 1]
            traj['advantages'] = adv
        else:
            raise NotImplementedError

    return avg_return / len(trajs)


def get_num_step2target(X, Y, bbox):
    """
    计算眼动轨迹中第一次注视到目标物体的步数

    参数:
    - X, Y: 眼动轨迹的坐标序列
    - bbox: 目标物体边界框 [x, y, width, height]

    返回:
    - 第一次注视目标的步数索引+1（从1开始计数）
    """
    X, Y = np.array(X), np.array(Y)

    # 1. 判断X坐标是否在目标区域内
    # bbox[0]: 左边界, bbox[0] + bbox[2]: 右边界
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])

    # 2. 判断Y坐标是否在目标区域内
    # bbox[1]: 上边界, bbox[1] + bbox[3]: 下边界
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])

    # 3. 同时满足X和Y都在目标区域内
    # on_target: 布尔数组，True表示该步注视点在目标区域内
    on_target = np.logical_and(on_target_X, on_target_Y)

    # 4. 检查是否曾经注视过目标
    if np.sum(on_target) > 0:
        # 找到第一次注视目标的索引位置
        first_on_target_idx = np.argmax(on_target)  # 第一个True的位置
        # 返回步数（从1开始计数，所以+1）
        return first_on_target_idx + 1
    else:
        # 如果从未注视过目标，返回一个大数值
        # 表示"搜索失败"或"搜索时间极长"
        return 1000  # 足够大的数字


def get_CDF(num_steps, max_step):
    """
    根据步数数组计算累积分布函数
    参数:
    - num_steps: 一个任务中所有试验的步数数组，如 [3, 5, 2, 8, 4]
    - max_step: 最大步数限制

    返回:
    - cdf: 累积分布函数数组，长度为max_step
    """
    # 初始化CDF数组，全为0
    cdf = np.zeros(max_step)

    # 总试验次数（用于计算百分比）
    total = float(len(num_steps))

    # 计算每一步的累积概率
    for i in range(1, max_step + 1):
        # 计算在第i步或之前找到目标的试验比例
        # np.sum(num_steps <= i): 统计步数≤i的试验个数
        cdf[i - 1] = np.sum(num_steps <= i) / total

    return cdf


def get_num_steps(trajs, target_annos, task_names):
    """
    计算每个任务中所有轨迹找到目标物体所需的步数

    参数:
    - trajs: 所有轨迹数据列表，每个轨迹包含眼动坐标序列
    - target_annos: 目标物体标注字典，格式 {任务名_图像名: 边界框}
    - task_names: 需要处理的任务名称列表

    返回:
    - num_steps: 字典，格式 {任务名: [步数数组]}
    """

    num_steps = {}  # 存储每个任务的步数统计

    # 遍历每个任务（如 'find_cup', 'find_phone' 等）
    for task in task_names:

        # 1. 筛选出当前任务的所有轨迹
        # 从所有轨迹中过滤出属于当前任务的轨迹
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))

        # 2. 初始化步数数组
        # 为当前任务的每个轨迹预分配步数存储空间，默认值为1
        # uint8: 无符号8位整数，节省内存（最大支持255步）
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)

        # 3. 遍历当前任务的每个轨迹
        for i, traj in enumerate(task_trajs):
            # 4. 构造标注键名
            # 格式: "任务名_图像名"，如 "find_cup_kitchen01"
            key = traj['task'] + '_' + traj['name']

            # 5. 获取目标物体的边界框信息
            # bbox: 目标物体在图像中的位置 [x_min, y_min, x_max, y_max]
            bbox = target_annos[key]

            # 6. 计算找到目标所需的步数
            # 分析眼动轨迹，确定在第几步注视点进入了目标区域
            # 返回值: 找到目标时的步数索引
            step_num = get_num_step2target(
                np.array(traj['X']),  # 眼动轨迹的X坐标序列
                np.array(traj['Y']),  # 眼动轨迹的Y坐标序列
                bbox  # 目标物体边界框
            )

            # 7. 记录该轨迹的步数
            num_steps_task[i] = step_num

            # 8. 截断轨迹到找到目标的步数
            # 只保留找到目标之前的轨迹部分，去除多余的后续步骤
            # 这样做是因为找到目标后的眼动行为不再属于"搜索"过程
            traj['X'] = traj['X'][:step_num]  # 截断X坐标
            traj['Y'] = traj['Y'][:step_num]  # 截断Y坐标

        # 9. 存储当前任务的所有步数
        num_steps[task] = num_steps_task

    return num_steps


def get_mean_cdf(num_steps, task_names, max_step):
    """
    计算每个任务的累积分布函数(CDF)

    参数:
    - num_steps: 字典，格式 {任务名: [步数数组]}
    - task_names: 任务名称列表
    - max_step: 最大步数

    返回:
    - cdf_tasks: 每个任务的CDF列表
    """
    cdf_tasks = []
    # 为每个任务计算其CDF曲线
    for task in task_names:
        # 获取当前任务的所有试验步数，计算该任务的CDF
        cdf_tasks.append(get_CDF(num_steps[task], max_step))

    return cdf_tasks


def compute_search_cdf(scanpaths, annos, max_step, return_by_task=False):
    """
    计算搜索累积分布函数(CDF) - 分析人类在视觉搜索中找到目标的概率随步数的变化

    参数:
    - scanpaths: 扫描路径数据列表，每个元素包含一个试验的眼动轨迹
    - annos: 目标物体的标注信息（边界框等）
    - max_step: 最大步数限制
    - return_by_task: 是否按任务分别返回结果

    返回:
    - 如果return_by_task=True: 返回每个任务的CDF字典
    - 如果return_by_task=False: 返回所有任务的平均CDF和标准差
    """

    # 1. 提取所有唯一的任务名称
    # 例如: ['find_cup', 'find_phone', 'find_keys']
    task_names = np.unique([traj['task'] for traj in scanpaths])

    # 2. 计算每个任务中找到目标所需的步数
    # num_steps: 字典，键为任务名，值为该任务下所有试验找到目标的步数列表
    # 例如: {'find_cup': [3, 5, 2, 8], 'find_phone': [4, 6, 3]}
    num_steps = get_num_steps(scanpaths, annos, task_names)

    # 3. 计算每个任务的平均CDF
    # cdf_tasks: 二维数组，每行代表一个任务的CDF
    # CDF[i] = 在第i步或之前找到目标的概率
    # 例如:
    # find_cup的CDF: [0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0]
    #                 步数: 0    1    2     3    4    5     6     7     8
    # 含义: 25%的人在第2步找到，50%在第3步前找到，75%在第5步前找到，100%在第8步前找到
    cdf_tasks = get_mean_cdf(num_steps, task_names, max_step + 1)

    # 4. 根据参数决定返回格式
    if return_by_task:
        # 返回每个任务单独的CDF
        # 格式: {'find_cup': [0.0, 0.0, 0.25, ...], 'find_phone': [...]}
        return dict(zip(task_names, cdf_tasks))
    else:
        # 返回所有任务的统计汇总
        # mean_cdf: 所有任务CDF的平均值 - 整体搜索难度
        # std_cdf: 所有任务CDF的标准差 - 任务间差异程度
        mean_cdf = np.mean(cdf_tasks, axis=0)  # 按列求平均（跨任务）
        std_cdf = np.std(cdf_tasks, axis=0)  # 按列求标准差

        return mean_cdf, std_cdf


def calc_overlap_ratio(bbox, patch_size, patch_num):
    """
    计算目标框 bbox 与每个 patch 的重叠比例（area of interest ratio）
    参数:
        bbox: [x, y, w, h]，目标框的左上角坐标(x, y)和宽高(w, h)
        patch_size: [patch_w, patch_h]，每个 patch 的大小
        patch_num: [num_rows, num_cols]，图像被分为多少个 patch（行数和列数）
    返回:
        aoi_ratio: shape 为 (1, num_cols, num_rows) 的数组，表示每个 patch 的重叠比例
    """
    # 每个 patch 的面积
    patch_area = float(patch_size[0] * patch_size[1])

    # 初始化输出数组，表示每个 patch 的重叠比例，初始为 0
    aoi_ratio = np.zeros((1, patch_num[1], patch_num[0]), dtype=np.float32)

    # bbox 左上角和右下角的坐标
    tl_x, tl_y = bbox[0], bbox[1]
    br_x, br_y = bbox[0] + bbox[2], bbox[1] + bbox[3]

    # bbox 在 patch 网格中的起始和终止 patch 索引
    lx, ux = tl_x // patch_size[0], br_x // patch_size[0]
    ly, uy = tl_y // patch_size[1], br_y // patch_size[1]

    # 遍历 bbox 所覆盖的所有 patch
    for x in range(lx, ux + 1):  # patch x 方向
        for y in range(ly, uy + 1):  # patch y 方向
            # 当前 patch 的左上角和右下角坐标
            patch_tlx, patch_tly = x * patch_size[0], y * patch_size[1]
            patch_brx, patch_bry = patch_tlx + patch_size[0], patch_tly + patch_size[1]

            # 求重叠区域的左上角坐标（取较大的起点）
            aoi_tlx = max(tl_x, patch_tlx)
            aoi_tly = max(tl_y, patch_tly)

            # 求重叠区域的右下角坐标（取较小的终点）
            aoi_brx = min(br_x, patch_brx)
            aoi_bry = min(br_y, patch_bry)

            # 计算重叠区域面积，除以 patch 面积得到比例（最多为 1）
            aoi_ratio[0, y, x] = max((aoi_brx - aoi_tlx), 0) * max((aoi_bry - aoi_tly), 0) / patch_area

    # 返回每个 patch 的重叠比例图
    return aoi_ratio



def get_center_keypoint_map(bbox, patch_num, box_size_dependent=True, normalize=True):
    xc, yc = np.round(bbox[0] + bbox[2] / 2), np.round(bbox[1] + bbox[3] / 2)
    if box_size_dependent:
        sigma = np.sqrt(bbox[2] ** 2 + bbox[3] ** 2) / 8
        target_map = np.zeros((320, 512), dtype=np.float32)
        target_map[int(yc), int(xc)] = 1
        target_map = filters.gaussian_filter(target_map, sigma=sigma)
        if patch_num[0] < 320:
            target_map = F.interpolate(
                torch.from_numpy(target_map).unsqueeze(0).unsqueeze(0),
                size=patch_num, mode='bilinear')
    else:
        target_map = np.zeros(patch_num, dtype=np.float32)
        target_map[int(yc // 16), int(xc // 16)] = 1
        target_map = filters.gaussian_filter(target_map, sigma=1)
        target_map = torch.from_numpy(target_map)

    if normalize:
        target_map /= target_map.sum()
    return target_map


def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    return mask.astype(np.float32)


def multi_hot_coding(bbox, patch_size, patch_num, thresh=0):
    """
    功能说明：
    将目标框（bbox）与图像中均匀划分的 patch 网格进行重叠面积计算，
    返回一个 multi-hot 编码向量，表示该目标框覆盖了哪些 patch 区域。
    参数:
        bbox: [x1, y1, x2, y2]，目标框坐标
        patch_size: [patch_w, patch_h]，每个 patch 的宽高
        patch_num: [num_rows, num_cols]，patch 的行数和列数（例如 [10, 16]）
        thresh: 重叠比例阈值（如果 bbox 和某 patch 的重叠比例大于该值就算“命中”）
    返回:
        aoi_ratio[0]：一个长度为 patch_num[0] × patch_num[1] 的 multi-hot 向量（0 或 1）
    """
    # 计算 bbox 与每个 patch 的重叠比例，返回形状为 (1, patch_num[0] * patch_num[1]) 的数组
    aoi_ratio = calc_overlap_ratio(bbox, patch_size, patch_num)

    # 标记哪些 patch 的重叠比例超过阈值（True 表示命中）
    hot_ind = aoi_ratio > thresh

    # 如果没有任何 patch 命中，就降低阈值直到至少命中一个
    while hot_ind.sum() == 0:
        thresh *= 0.5                    # 阈值减半
        hot_ind = aoi_ratio > thresh     # 重新判断命中情况

    # 将命中的 patch 设为 1，表示该位置被 bbox 覆盖
    aoi_ratio[hot_ind] = 1

    # 将未命中的 patch 设为 0
    aoi_ratio[np.logical_not(hot_ind)] = 0

    # 返回最终的 multi-hot 向量（只取第一个维度）
    return aoi_ratio[0]



def actions2scanpaths(actions_all, patch_num):
    # convert actions to scanpaths
    scanpaths = []
    for traj in actions_all:
        task_name, img_name, condition, actions = traj
        actions = actions.to(dtype=torch.float32)
        if actions[-1] == patch_num[0] * patch_num[1]:
            actions = actions[:-1]  # remove stopping action
        py = (actions // patch_num[0]) / float(patch_num[1])
        px = (actions % patch_num[0]) / float(patch_num[0])
        fixs = torch.stack([px, py])
        fixs = np.concatenate([np.array([[0.5], [0.5]]),
                               fixs.cpu().numpy()],
                              axis=1)
        scanpaths.append({
            'X': fixs[0] * 512,
            'Y': fixs[1] * 320,
            'name': img_name,
            'task': task_name,
            'condition': condition
        })
    return scanpaths


def preprocess_fixations(trajs,
                         patch_size,
                         patch_num,
                         im_h,
                         im_w,
                         truncate_num=-1,
                         need_label=True,
                         has_stop=False,
                         sample_scanpath=False,
                         min_traj_length_percentage=0,
                         discretize_fix=True,
                         remove_return_fixations=False,
                         is_coco_dataset=True):
    """
    眼动轨迹预处理函数 - 将原始眼动数据转换为模型训练格式

    参数详解:
    - trajs: 原始轨迹数据列表，每个轨迹包含X,Y坐标序列和时间信息
    - patch_size: 每个patch的像素大小 [patch_w, patch_h]
    - patch_num: patch网格数量 [num_x, num_y]
    - im_h, im_w: 图像尺寸
    - truncate_num: 轨迹截断长度(-1表示不截断)
    - need_label: 是否需要标签(未使用)
    - has_stop: 是否添加停止动作
    - sample_scanpath: 是否采样扫描路径
    - min_traj_length_percentage: 最小轨迹长度百分比
    - discretize_fix: 是否离散化注视点坐标
    - remove_return_fixations: 是否移除返回注视(抑制返回机制)
    - is_coco_dataset: 是否为COCO数据集

    返回:
    - fix_labels: 处理后的注视点标签列表
    """

    fix_labels = []  # 存储所有处理后的注视点标签

    # 空轨迹检查
    if len(trajs) == 0:
        return fix_labels

    # 检查是否包含扫描路径ID信息
    has_spid = 'scanpath_id' in trajs[0].keys()

    # 遍历每个轨迹进行处理
    for traj in trajs:

        # ==================== 1. 初始注视点处理 ====================
        if is_coco_dataset:
            # COCO数据集：强制将初始注视点设为图像中心
            # 这是一个标准化操作，确保所有COCO轨迹都从中心开始
            traj['X'][0], traj['Y'][0] = im_w // 2, im_h // 2

        # 将初始注视点坐标转换为离散动作ID
        discrete_label = pos_to_action(
            traj['X'][0], traj['Y'][0], patch_size, patch_num)

        # 根据是否离散化选择不同的处理方式
        if discretize_fix:
            # 离散化模式：使用patch网格
            label = discrete_label
            # 将动作ID转回到patch中心坐标
            tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
            fixs = [(tar_x, tar_y)]  # 存储离散化后的坐标
        else:
            # 连续模式：保持原始像素坐标
            label = pos_to_action(
                traj['X'][0], traj['Y'][0], [1, 1], [im_w, im_h])
            fixs = [(traj['X'][0], traj['Y'][0])]  # 存储原始坐标

        # 初始化标签历史(用于抑制返回机制)
        label_his = [discrete_label]

        # ==================== 2. 轨迹长度设置 ====================
        if truncate_num < 1:
            # 不截断，使用完整轨迹长度
            traj_len = len(traj['X'])
        else:
            # 截断到指定长度
            traj_len = min(truncate_num, len(traj['X']))

        # 计算最小轨迹长度阈值
        min_traj_length = int(min_traj_length_percentage * traj_len)

        # ==================== 4. 处理后续注视点 ====================
        for i in range(1, traj_len):

            # 4.1 边界检查：移除越界的注视点
            if (traj['X'][i] >= im_w or traj['Y'][i] >= im_h or
                    traj['X'][i] < 0 or traj['Y'][i] < 0):
                continue

            # 4.2 坐标转换：像素坐标 → 动作ID
            discrete_label = pos_to_action(
                traj['X'][i], traj['Y'][i], patch_size, patch_num)

            if discretize_fix:
                label = discrete_label
            else:
                label = pos_to_action(
                    traj['X'][i], traj['Y'][i], [1, 1], [im_w, im_h])

            # 4.3 抑制返回机制：移除重复访问的位置
            if remove_return_fixations and discrete_label in label_his:
                continue

            # 4.4 更新访问历史
            label_his.append(discrete_label)

            # 4.5 构建注视点标签
            # 标签格式: [图像名, 任务, 条件, 历史注视点, 当前标签, 是否停止, 被试ID, (扫描路径ID), 累积时间]
            if has_spid:
                fix_label = [traj['name'], traj['task'], traj['condition'],
                             copy(fixs), label, False, traj['subject'],
                             traj['scanpath_id'], np.sum(traj['T'][:i])]
            else:
                fix_label = [traj['name'], traj['task'], traj['condition'],
                             copy(fixs), label, False, traj['subject'],
                             np.sum(traj['T'][:i])]

            # 4.6 更新注视点历史
            if discretize_fix:
                # 离散化：添加patch中心坐标
                tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
                fixs.append((tar_x, tar_y))
            else:
                # 连续：添加原始坐标
                fixs.append((traj['X'][i], traj['Y'][i]))

            # 4.7 添加到结果列表
            if (not sample_scanpath) and i >= min_traj_length:
                fix_labels.append(fix_label)

        # ==================== 5. 添加停止动作 ====================
        if has_stop or sample_scanpath:
            # 在扫描路径末尾添加停止动作
            # 停止动作的标签ID = patch总数 (表示"停止搜索")
            if has_spid:
                stop_label = [traj['name'], traj['task'], traj['condition'],
                              copy(fixs), patch_num[0] * patch_num[1], True,
                              traj['subject'], traj['scanpath_id'],
                              np.sum(traj['T'])]
            else:
                stop_label = [traj['name'], traj['task'], traj['condition'],
                              copy(fixs), patch_num[0] * patch_num[1], True,
                              traj['subject'], np.sum(traj['T'])]
            fix_labels.append(stop_label)

    return fix_labels


def _file_at_step(step, name):
    return "save_{}_{}k{}.pkg".format(name, int(step // 1000),
                                      int(step % 1000))


def _file_best(name):
    return "trained_{}.pkg".format(name)


def save(global_step,
         model,
         optim,
         name,
         pkg_dir="",
         is_best=False,
         max_checkpoints=None):
    if optim is None:
        raise ValueError("cannot save without optimzier")
    state = {
        "global_step":
            global_step,
        # DataParallel wrap model in attr `module`.
        "model":
            model.module.state_dict()
            if hasattr(model, "module") else model.state_dict(),
        "optim":
            optim.state_dict(),
    }
    save_path = os.path.join(pkg_dir, _file_at_step(global_step, name))
    best_path = os.path.join(pkg_dir, _file_best(name))
    torch.save(state, save_path)
    print("[Checkpoint]: save to {} successfully".format(save_path))

    if is_best:
        copyfile(save_path, best_path)
    if max_checkpoints is not None:
        history = []
        for file_name in os.listdir(pkg_dir):
            if re.search("save_{}_\d*k\d*\.pkg".format(name), file_name):
                digits = file_name.replace("save_{}_".format(name),
                                           "").replace(".pkg", "").split("k")
                number = int(digits[0]) * 1000 + int(digits[1])
                history.append(number)
        history.sort()
        while len(history) > max_checkpoints:
            path = os.path.join(pkg_dir, _file_at_step(history[0], name))
            print("[Checkpoint]: remove {} to keep {} checkpoints".format(
                path, max_checkpoints))
            if os.path.exists(path):
                os.remove(path)
            history.pop(0)


def load(step_or_path, model, name, optim=None, pkg_dir="", device=None):
    step = step_or_path
    save_path = None
    if isinstance(step, int):
        save_path = os.path.join(pkg_dir, _file_at_step(step, name))
    if isinstance(step, str):
        if pkg_dir is not None:
            if step == "best":
                save_path = os.path.join(pkg_dir, _file_best(name))
            else:
                save_path = os.path.join(pkg_dir, step)
        else:
            save_path = step
    if save_path is not None and not os.path.exists(save_path):
        print("[Checkpoint]: Failed to find {}".format(save_path))
        return
    if save_path is None:
        print("[Checkpoint]: Cannot load the checkpoint")
        return

    # begin to load
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    model.load_state_dict(state["model"])
    if optim is not None:
        optim.load_state_dict(state["optim"])

    print("[Checkpoint]: Load {} successfully".format(save_path))
    return global_step


def genGaussiankernel(width, sigma):
    x = np.arange(-int(width / 2), int(width / 2) + 1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d


def pyramid(im, sigma=1, prNum=6, transform=None):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]

    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)

    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width / 2), int(height / 2)))
        pyramids.append(G)

    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            curr_im = cv2.resize(curr_im,
                                 (curr_im.shape[1] * 2, curr_im.shape[0] * 2))
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids


def foveat_img(im, fixs, As=None):
    sigma = 0.248
    prNum = 6
    if As is None:
        As = pyramid(im, sigma, prNum)
        height, width, _ = im.shape
    else:
        height, width, _ = As[0].shape

    # compute coef
    p = 7.5  # 16
    k = 1.5  # 1.02
    alpha = 1.5

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta,
                           np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)

    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i - 3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i - 1] = np.sqrt(np.log(2) / k) / (2 ** (i - 3)) * sigma

    omega[omega > 1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i - 1] - Ts[i] + 1e-5))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))
    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i - 1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov, Ms


def real_foveation_batch(As, fixation_batch, pa):
    """
    This function generate foveated image in batch on GPU

    **As**: batch of image pyrimaids of the shape (batch_size,
    pyrNum, channel, height, width).

    **fixation_batch**: (batch_size, fix_num, 2) tensor in (x, y)
    """
    sigma = 0.248
    prNum = 6
    width = pa.im_w
    height = pa.im_h
    patch_size = pa.patch_size
    device = As.device

    prNum = As.size()[1]
    batch_size = As.size()[0]
    fix_num = fixation_batch.size(1)

    # compute coef
    p = 7.5
    k = 1.5
    alpha = 2.5

    x = torch.arange(0, width, device=device, dtype=torch.float)
    y = torch.arange(0, height, device=device, dtype=torch.float)
    y2d, x2d = torch.meshgrid([y, x])
    h, w = x2d.size()

    x2d = x2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)
    y2d = y2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)

    # fixation patch index to fixation pixel coordinates
    xc = fixation_batch[:, :, 0] * patch_size[0] + patch_size[0] / 2
    yc = fixation_batch[:, :, 1] * patch_size[1] + patch_size[1] / 2

    xc2d = xc.view(batch_size, fix_num, 1, 1).expand_as(x2d)
    yc2d = yc.view(batch_size, fix_num, 1, 1).expand_as(y2d)

    theta = torch.sqrt((x2d - xc2d) ** 2 + (y2d - yc2d) ** 2) / p
    theta, _ = torch.min(theta, dim=1)
    R = alpha / (theta + alpha)

    Ts = torch.zeros((batch_size, 6, height, width), device=device)
    for i in range(prNum - 1):
        Ts[:, i] = torch.exp(-((2 ** (i - 2)) * R / sigma) ** 2 * k)

    # omega
    omega = np.zeros(prNum)
    omega[:-1] = math.sqrt(math.log(2) / k) / (2 ** np.arange(
        -2, 3, dtype=float)) * sigma
    omega[omega > 1] = 1

    # layer index
    layer_ind = torch.zeros_like(R, device=device)
    for i in range(1, prNum):
        ind = (R >= omega[i]) * (R <= omega[i - 1])
        layer_ind[ind] = i

    # Bs
    Bs = (0.5 - Ts[:, 1:]) / (Ts[:, :-1] - Ts[:, 1:])

    # M
    Ms = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum):
        ind = layer_ind == i
        if torch.sum(ind) > 0:
            if i == 0:
                Ms[:, i][ind] = 1
            else:
                Ms[:, i][ind] = 1 - Bs[:, i - 1][ind]

        ind = layer_ind - 1 == i
        if torch.sum(ind) > 0:
            Ms[:, i][ind] = Bs[:, i][ind]

    # generate periphery image
    Ms = Ms.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    im_fov_batch = torch.sum(Ms * As, dim=1)

    return im_fov_batch, R


# Convert discrete fixation dataset to continuous density map
def convert_fixations_to_map(fixs,
                             width,
                             height,
                             return_distribution=True,
                             smooth=True,
                             visual_angle=16):
    """
    将注视点坐标列表转换为注视热力图（可平滑、归一化为分布）

    参数说明:
    - fixs: ndarray 或 list，形状为 [N, 2] 的注视点坐标 (x, y)，像素单位
    - width: int，目标图像的宽度（如512）
    - height: int，目标图像的高度（如320）
    - return_distribution: bool，是否将结果归一化为概率分布（总和为1）
    - smooth: bool，是否对图进行高斯平滑处理（模拟视野模糊）
    - visual_angle: float，高斯核的标准差，控制模糊程度（越大越平滑）

    返回:
    - fmap: ndarray [height, width]，注视热力图或概率分布图
    """

    # 边界检查：确保至少有一个注视点
    assert len(fixs) > 0, 'Empty fixation list!'  # 否则报错

    # 初始化全零热力图（大小与图像相同）
    fmap = np.zeros((height, width))

    # 遍历所有注视点，将其在热力图中位置对应的像素值加1
    for i in range(len(fixs)):
        x, y = fixs[i][0], fixs[i][1]  # 提取当前注视点坐标
        fmap[y, x] += 1                # 注意：行是y，列是x（矩阵访问方式）

    # 是否进行高斯模糊（模拟人眼的感知模糊）
    if smooth:
        # 对整个热力图进行高斯滤波处理
        # visual_angle 控制模糊程度，相当于 sigma
        fmap = filters.gaussian_filter(fmap, sigma=visual_angle)

    # 是否将结果归一化为概率分布（即所有像素值之和为1）
    if return_distribution:
        fmap /= fmap.sum()  # 归一化：将热力图变为概率图

    # 返回最终的注视图（或概率图）
    return fmap

def get_prior_maps(gt_scanpaths, im_w, im_h, visual_angle=24):
    """
    生成先验概率图 - 统计人类在不同搜索任务中的注视行为模式

    参数:
    - gt_scanpaths: 真实扫描路径数据列表
    - im_w, im_h: 图像尺寸 (通常512x320)
    - visual_angle: 视觉角度，影响高斯平滑程度，默认24度

    返回:
    - prior_maps: 字典，包含每个搜索任务的先验概率图
    """

    # 边界条件：如果没有扫描路径数据，返回空字典
    if len(gt_scanpaths) == 0:
        return {}

    # 提取所有唯一的搜索任务名称
    # 例如：['bottle', 'person', 'car', 'cup', ...] (COCO-Search18的18个类别)
    task_names = np.unique([traj['task'] for traj in gt_scanpaths])

    # 初始化存储结构
    all_fixs = []  # 存储所有任务的注视点，用于生成总体先验图
    prior_maps = {}  # 最终返回的先验概率图字典

    # 遍历每个搜索任务，分别统计其注视模式
    for task in task_names:
        # 收集该任务在训练集中的所有X坐标
        # 关键：使用traj['X'][1:]跳过初始注视点（通常是图像中心，不反映搜索策略）
        # 只包含训练集数据（split=='train'）和当前任务的轨迹
        Xs = np.concatenate([
            traj['X'][1:]  # [1:] 跳过第一个注视点，从真正的搜索行为开始
            for traj in gt_scanpaths
            if traj['split'] == 'train' and traj['task'] == task  # 筛选条件
        ])

        # 收集该任务在训练集中的所有Y坐标（与X坐标对应）
        Ys = np.concatenate([
            traj['Y'][1:]  # 同样跳过初始注视点
            for traj in gt_scanpaths
            if traj['split'] == 'train' and traj['task'] == task
        ])

        # 将X,Y坐标组合成注视点数组 [N, 2]，N为该任务的总注视点数
        # .T 转置操作：从[2, N]变为[N, 2]格式
        # astype(np.int32) 转换为整数坐标（像素位置）
        fixs = np.stack([Xs, Ys]).T.astype(np.int32)

        # 将该任务的注视点转换为先验概率图
        # convert_fixations_to_map函数会：
        # 1. 创建空白热力图 [im_h, im_w]
        # 2. 在每个注视点位置累加计数
        # 3. 应用高斯平滑（visual_angle控制平滑程度）
        # 4. 归一化为概率分布
        prior_maps[task] = convert_fixations_to_map(fixs,
                                                    im_w,  # 图像宽度 512
                                                    im_h,  # 图像高度 320
                                                    smooth=True,  # 启用高斯平滑
                                                    visual_angle=visual_angle)  # 24度视觉角

        # 保存该任务的注视点，后续用于生成包含所有任务的综合先验图
        all_fixs.append(fixs)

    # 合并所有任务的注视点，生成通用的先验概率图
    # 这个'all'先验图反映了人类视觉搜索的通用模式，不特定于某个目标
    all_fixs = np.concatenate(all_fixs)  # 将所有任务的注视点数组拼接

    # 生成综合先验图，包含所有搜索任务的注视行为
    prior_maps['all'] = convert_fixations_to_map(all_fixs,
                                                 im_w,
                                                 im_h,
                                                 smooth=True,
                                                 visual_angle=visual_angle)

    # 返回包含各任务先验图的字典
    # 结构：{'bottle': 2D数组, 'person': 2D数组, ..., 'all': 2D数组}
    # 每个2D数组的值表示该位置被注视的概率密度
    return prior_maps


def get_IoM(bb1, bb2):
    """
    计算两个边界框的 IoM（Intersection over Minimum）值。

    参数：
        bb1: dict，表示第一个框，包含键 {'x1', 'y1', 'x2', 'y2'}
        bb2: dict，表示第二个框，格式同上

    返回：
        float，表示两个框的交集面积与两者中较小面积的比值（范围为[0,1]）
    """

    # 保证输入的两个框是合法的（左上 < 右下）
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # 计算交集区域的左上角与右下角坐标（取重叠部分）
    x_left = max(bb1['x1'], bb2['x1'])    # 左边界：两个框中较靠右的左边界
    y_top = max(bb1['y1'], bb2['y1'])     # 上边界：两个框中较靠下的上边界
    x_right = min(bb1['x2'], bb2['x2'])   # 右边界：两个框中较靠左的右边界
    y_bottom = min(bb1['y2'], bb2['y2'])  # 下边界：两个框中较靠上的下边界

    # 如果无重叠区域，交集面积为 0，IoM 也为 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个框的面积
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # 计算 IoM = 交集 / 较小框的面积
    iom = intersection_area / float(min(bb1_area, bb2_area))

    # 将结果限制在 [0, 1] 范围内，避免浮点误差
    iom = np.clip(iom, 0, 1)

    return iom



def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(
        mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def transform_fixations(normalized_fixations, is_padding, pa, sample_scanpath=True, return_highres=False):
    """
    将归一化注视点序列转换为分类标签序列（即 1D patch 索引），
    可用于训练或预测模型中关注位置的分类任务。
    参数：
        normalized_fixations: Tensor，形状为 (B, T, 2)，表示一批注视点序列（坐标值归一化到 0~1）
        is_padding: Tensor，形状为 (B, T)，布尔值表示每个时间步是否为 padding（可为 None）
        pa: 参数对象，包含图像尺寸、终止索引、填充索引等信息
        sample_scanpath: 是否为采样阶段（True 表示训练中的采样过程，需要设置终止标签）
        return_highres: 是否返回高分辨率的标签序列（True 则同时返回高分辨率版本）
    返回：
        fix_seq: 低分辨率分类标签序列（patch 索引）
        fix_seq_high: 高分辨率版本（可选返回）
    """
    def transform(normalized_fixations, is_padding, patch_num, sample_scanpath):
        """
        将归一化注视点转换为 patch 索引（标签）
        参数：
            normalized_fixations: (B, T, 2)，归一化坐标
            is_padding: (B, T)，padding 掩码
            patch_num: [宽patch数, 高patch数]
            sample_scanpath: 是否为采样过程（需要加终止标记）
        返回：
            labels: LongTensor，patch 序列标签 (B, T)
        """
        # 将归一化注视点映射到 patch 网格索引
        fixs = (normalized_fixations * torch.Tensor(patch_num)).to(torch.long)

        # 将 2D patch 索引转换为 1D 标签 这是在把二维注视点 (x, y) 位置，转换成一个一维的序列标签（index）。
        # 横坐标作为低位，纵坐标作为高位（行优先）
        labels = patch_num[0] * fixs[:, :, 1] + fixs[:, :, 0]

        # 所有标签统一偏移（+1 或 +2），防止与 pad_idx 冲突
        labels += 1 + int(sample_scanpath)

        # 如果是 padding 的位置，则赋值为 pad_idx
        # 有些样本的注视点数量少于最大长度（比如只看了 3 次，而你允许最多 10 次）
        # 那么就需要对这些“补齐的位置”打上 pad_idx 标记
        # 这样模型在训练时就可以跳过这些 padding 区域，不去计算损失或注意力
        labels[is_padding == 1] = pa.pad_idx

        # 如果是采样过程，还需要设置终止符号（eos_idx）
        if sample_scanpath:
            # 找出每条序列的 padding 开始位置（即真实结束位置） 关键规则：dim=n 表示减第n维，保留其他维度。
            term_idx = is_padding.argmax(dim=1)

            # 将终止位置设置为 eos_idx（如果这条序列有 padding）  布尔索引语法‘’
            # [term_idx > 0] 保留那些 mask 为 True 的位置（只保留相应的 index）。
            labels[torch.arange(len(labels))[term_idx > 0], term_idx[term_idx > 0]] = pa.eos_idx

        return labels.to(torch.long)


    """
    为什么 patch 的大小会影响精度？它和模型处理视觉信息的能力之间有什么联系？
    patch_num = [pa.im_w // 32, pa.im_h // 32]  # 低分辨率
    patch_num = [pa.im_w // 4, pa.im_h // 4]    # 高分辨率
    它的含义是：
    把图像切成网格，每个格子是一个 patch，比如：
        图像大小为 256×256
        那么：
            低分辨率划分为 32×32 patch，大概有 (256/32)^2 = 64 个 patch
            高分辨率划分为 4×4 patch，大概有 (256/4)^2 = 4096 个 patch
    每个 patch 最终会变成一个 token：相当于「注视区域的离散编号」。
    
    为什么 patch 越小，精度越高？
    低分辨率（32×32 patch）时，一个 patch = 32×32 像素：
        你注视点落在 (45, 60)，它和 (60, 70) 都会映射成同一个 patch（比如编号 10），区别不了它们。
    高分辨率（4×4 patch）时，一个 patch = 4×4 像素：
        注视点 (45, 60) 和 (60, 70) 就会变成不同的 patch index，能精确区分每个注视点的位置差异。
🧠 3. 为什么需要两种分辨率？
    高分辨率和低分辨率是各有优缺的：
    分辨率	优点	缺点
    高分辨率（小 patch）	更精细、拟人更真实、区分更强	token 太多、计算成本高
    低分辨率（大 patch）	快速、粗略捕捉趋势、适合全局处理	模糊、不够精细、可能混淆位置
    所以模型中通常这样用：
        低分辨率 token：用于全局 transformer 编码（粗略位置感知）
        高分辨率 token：用于细节预测、termination 判断、细粒度行为模拟
    """
    fix_seq = transform(normalized_fixations, is_padding,
                        [pa.im_w // 32, pa.im_h // 32], sample_scanpath)

    if return_highres:
        # 高分辨率转换（每 4 像素为一个 patch，更细粒度）
        fix_seq_high = transform(normalized_fixations, is_padding,
                                 [pa.im_w // 4, pa.im_h // 4], sample_scanpath)
        return fix_seq, fix_seq_high
    else:
        return fix_seq, None


def create_mask(tgt, pad_idx, device):
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return tgt_mask, tgt_padding_mask



