import numpy as np
from os.path import join
from torchvision import transforms
from .utils import compute_search_cdf, preprocess_fixations,cutFixOnTarget
from .data import FFN_IRL, GLDAS_Human_Gaze
def process_data(target_trajs, dataset_root, target_annos, hparams, target_trajs_all,
                 is_testing=False, sample_scanpath=False, min_traj_length_percentage=0,
                 use_coco_annotation=False):
    print("using", hparams.Train.repr, 'dataset:', hparams.Data.name, 'TAP:', hparams.Data.TAP)
    coco_annos = None
    if  hparams.Data.name=="COCO-Search18":
        ori_h, ori_w = 320, 512
        rescale_flag = hparams.Data.im_h != ori_h
    else:
        print(f"dataset {hparams.Data.name} not supported")
        raise NotImplementedError

    if rescale_flag:
        print(f"Rescaling image and fixation to {hparams.Data.im_h}x{hparams.Data.im_w}")
        ratio_h = hparams.Data.im_h / ori_h
        ratio_w = hparams.Data.im_w / ori_w
        for traj in target_trajs_all:
            traj['X'] = np.array(traj['X']) * ratio_w
            traj['Y'] = np.array(traj['Y']) * ratio_h
            traj['rescaled'] = True

    if hparams.Train.repr == 'FFN':
        size = (hparams.Data.im_h, hparams.Data.im_w)
        transform_train = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError
    valid_target_trajs_all = list(
        filter(lambda x: x['split'] == 'test', target_trajs_all))

    fix_clusters = np.load(f'{dataset_root}/clusters.npy',allow_pickle=True).item()
    for _, v in fix_clusters.items():
        if isinstance(v['strings'], list):
            break
        if hparams.Data.subject > -1:
            try:
                v['strings'] = [v['strings'][hparams.Data.subject]]
            except:
                v['strings'] = []
        else:
            v['strings'] = list(v['strings'].values())

    is_coco_dataset = hparams.Data.name == 'COCO-Search18'
    if is_coco_dataset:
        scene_labels = np.load(f'{dataset_root}/scene_label_dict.npy',
                               allow_pickle=True).item()
    else:
        scene_labels = None

    target_init_fixs = {}
    for traj in target_trajs_all:
        key = traj['task'] + '*' + traj['name'] + '*' + traj['condition']
        if is_coco_dataset:
            target_init_fixs[key] = (0.5, 0.5)
        else:
            target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                     traj['Y'][0] / hparams.Data.im_h)

    if hparams.Train.zero_shot:
        catIds = np.load(join(dataset_root, 'all_task_ids.npy'),
                         allow_pickle=True).item()
    else:
        cat_names = list(np.unique([x['task'] for x in target_trajs]))
        catIds = dict(zip(cat_names, list(range(len(cat_names)))))
    human_mean_cdf = None
    if is_testing:
        test_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        assert len(test_target_trajs) > 0, 'no testing dataset found!'

        test_task_img_pair = np.unique([
            traj['task'] + '*' + traj['name'] + '*' + traj['condition']
            for traj in test_target_trajs
        ])

        traj_lens = list(map(lambda x: x['length'], test_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of train trajs = {}'.format(len(test_target_trajs)))

        if hparams.Data.TAP == 'TP':
            human_mean_cdf, _ = compute_search_cdf(
                test_target_trajs, target_annos, hparams.Data.max_traj_length)
            print('target fixation prob (test).:', human_mean_cdf)

        if hparams.Train.repr == 'FFN':
            test_img_dataset = FFN_IRL(dataset_root, target_init_fixs,
                                       test_task_img_pair, target_annos,
                                       transform_test, hparams.Data, catIds)

        return {
            'catIds': catIds,
            'img_test': test_img_dataset,
            'bbox_annos': target_annos,
            'gt_scanpaths': test_target_trajs,
            'fix_clusters': fix_clusters
        }

    else:
        train_target_trajs = list(
            filter(lambda x: x['split'] == 'train', target_trajs)        )

        traj_lens = list(map(lambda x: x['length'], train_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of train trajs = {}'.format(len(train_target_trajs)))

        train_task_img_pair = np.unique([
            traj['task'] + '*' + traj['name'] + '*' + traj['condition']
            for traj in train_target_trajs
        ])

        train_fix_labels = preprocess_fixations(
            train_target_trajs,
            hparams.Data.patch_size, hparams.Data.patch_num,
            hparams.Data.im_h, hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )
        valid_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs)        )

        traj_lens = list(map(lambda x: x['length'], valid_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average valid scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of valid trajs = {}'.format(len(valid_target_trajs)))

        if hparams.Data.TAP in ['TP', 'TAP']:
            tp_trajs = list(
                filter(
                    lambda x: x['condition'] == 'present' and x['split'] == 'test',
                    target_trajs_all))
            human_mean_cdf, _ = compute_search_cdf(
                tp_trajs, target_annos, hparams.Data.max_traj_length)
            print('target fixation prob (valid).:', human_mean_cdf)

        valid_fix_labels = preprocess_fixations(
            valid_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )

        valid_target_trajs_TP = list(
            filter(lambda x: x['condition'] == 'present',
                   valid_target_trajs_all))
        valid_fix_labels_TP = preprocess_fixations(
            valid_target_trajs_TP,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            is_coco_dataset=is_coco_dataset,
        )

        valid_target_trajs_TA = list(
            filter(lambda x: x['condition'] == 'absent',
                   valid_target_trajs_all))
        valid_fix_labels_TA = preprocess_fixations(
            valid_target_trajs_TA,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )

        valid_task_img_pair_TP = np.unique([
            traj['task'] + '*' + traj['name'] + '*' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'present'
        ])
        valid_task_img_pair_TA = np.unique([
            traj['task'] + '*' + traj['name'] + '*' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'absent'
        ])
        valid_task_img_pair_all = np.unique([
            traj['task'] + '*' + traj['name'] + '*' + traj['condition']
            for traj in valid_target_trajs_all
        ])

        if hparams.Train.repr == 'FFN':
            train_img_dataset = FFN_IRL(dataset_root, None,
                                        train_task_img_pair, target_annos,
                                        transform_train, hparams.Data, catIds,
                                        coco_annos=coco_annos)
            valid_img_dataset_all = FFN_IRL(dataset_root, None,
                                            valid_task_img_pair_all, target_annos,
                                            transform_test, hparams.Data, catIds,
                                            coco_annos=None)
            valid_img_dataset_TP = FFN_IRL(dataset_root, None,
                                           valid_task_img_pair_TP, target_annos,
                                           transform_test, hparams.Data,
                                           catIds, coco_annos=None)
            valid_img_dataset_TA = FFN_IRL(dataset_root, None,
                                           valid_task_img_pair_TA,
                                           target_annos, transform_test,
                                           hparams.Data, catIds, None)
            gaze_dataset_func = GLDAS_Human_Gaze
            train_HG_dataset = gaze_dataset_func(dataset_root,
                                                 train_fix_labels,
                                                 target_annos,
                                                 scene_labels,
                                                 hparams.Data,
                                                 transform_train,
                                                 catIds,
                                                 blur_action=True,
                                                 coco_annos=coco_annos)
            valid_HG_dataset = gaze_dataset_func(dataset_root,
                                                 valid_fix_labels,
                                                 target_annos,
                                                 scene_labels,
                                                 hparams.Data,
                                                 transform_test,
                                                 catIds,
                                                 blur_action=True,
                                                 coco_annos=None)
            valid_HG_dataset_TP = gaze_dataset_func(dataset_root,
                                                    valid_fix_labels_TP,
                                                    target_annos,
                                                    scene_labels,
                                                    hparams.Data,
                                                    transform_test,
                                                    catIds,
                                                    blur_action=True,
                                                    coco_annos=None)
            valid_HG_dataset_TA = gaze_dataset_func(dataset_root,
                                                    valid_fix_labels_TA,
                                                    target_annos,
                                                    scene_labels,
                                                    hparams.Data,
                                                    transform_test,
                                                    catIds,
                                                    blur_action=True,
                                                    coco_annos=None)

        if hparams.Data.TAP == ['TP']:
            cutFixOnTarget(target_trajs, target_annos)

        print("num of training and eval fixations = {}, {}".format(
            len(train_HG_dataset), len(valid_HG_dataset)))
        print("num of training and eval images = {}, {} (TP), {} (TA)".
              format(len(train_img_dataset), len(valid_img_dataset_TP),
                     len(valid_img_dataset_TA)))

        return {
            'catIds': catIds,
            'img_train': train_img_dataset,
            'img_valid_TP': valid_img_dataset_TP,
            'img_valid_TA': valid_img_dataset_TA,
            'img_valid': valid_img_dataset_all,
            'gaze_train': train_HG_dataset,
            'gaze_valid': valid_HG_dataset,
            'gaze_valid_TP': valid_HG_dataset_TP,
            'gaze_valid_TA': valid_HG_dataset_TA,
            'bbox_annos': target_annos,
            'fix_clusters': fix_clusters,
            'valid_scanpaths': valid_target_trajs_all,
            'human_cdf': human_mean_cdf,
        }