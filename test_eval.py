import argparse
import os
import torch

from gldas_gaze.builder import build
from common.config import JsonConfig
from gldas_gaze.evaluation import evaluate


def parse_args():
    parser = argparse.ArgumentParser("Evaluate a trained GLDAS-GAZE model")
    parser.add_argument("--hparams", type=str, required=True,
                        help="Path to the main hparams JSON")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pt file to evaluate")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-mode", choices=["greedy", "sample"], default="greedy",
                        help="Scanpath decoding at eval time")
    parser.add_argument("--disable-saliency", action="store_true",
                        help="Disable saliency metrics during eval")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--tap", choices=["TP", "TA", "ALL"], default="ALL",
                        help="Which task(s) to evaluate: TP, TA, or ALL.")
    return parser.parse_args()


def run_evaluation(model, device, tap_flag,
                   valid_img_loader_tp, valid_img_loader_ta,
                   hparams_tp, hparams_ta,
                   bbox_annos, human_cdf, fix_clusters,
                   prior_maps_tp, prior_maps_ta,
                   sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta,
                   sps_test_tp, sps_test_ta,
                   dataset_root, sample_action, output_saliency_metrics,
                   center_initial, log_dir):
    rst_tp = rst_ta = None

    if tap_flag in ["TP", "ALL"]:
        rst_tp, _ = evaluate(
            model, device,
            valid_img_loader_tp, valid_gaze_loader_tp,
            hparams_tp.Data,
            bbox_annos, human_cdf, fix_clusters,
            prior_maps_tp, sss_strings, dataset_root, sps_test_tp,
            sample_action=sample_action,
            output_saliency_metrics=output_saliency_metrics,
            center_initial=center_initial,
            log_dir=log_dir
        )
        print("TP:", rst_tp)

    if tap_flag in ["TA", "ALL"]:
        rst_ta, _ = evaluate(
            model, device,
            valid_img_loader_ta, valid_gaze_loader_ta,
            hparams_ta.Data,
            bbox_annos, human_cdf, fix_clusters,
            prior_maps_ta, sss_strings, dataset_root, sps_test_ta,
            sample_action=sample_action,
            output_saliency_metrics=output_saliency_metrics,
            center_initial=center_initial,
            log_dir=log_dir
        )
        print("TA:", rst_ta)

    return rst_tp, rst_ta


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    hparams = JsonConfig(args.hparams)
    cfg_dir = os.path.dirname(args.hparams)
    hparams_tp = JsonConfig(os.path.join(cfg_dir, 'coco_search18_dense_SSL_TP.json'))
    hparams_ta = JsonConfig(os.path.join(cfg_dir, 'coco_search18_dense_SSL_TA.json'))

    dataset_root = args.dataset_root.rstrip('/')
    output_saliency_metrics = not args.disable_saliency
    sample_action = args.eval_mode == 'sample'

    model, optimizer, train_gaze_loader, val_gaze_loader, train_img_loader, \
        valid_img_loader_tp, valid_img_loader_ta, \
        global_step, bbox_annos, human_cdf, fix_clusters, prior_maps_tp, \
        prior_maps_ta, sss_strings, valid_gaze_loader_tp, \
        valid_gaze_loader_ta, sps_test_tp, \
        sps_test_ta, term_pos_weight, _ = build(
        hparams, dataset_root, device, is_pretraining=False, is_eval=True, split=args.split)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    log_dir = hparams.Train.log_dir if hasattr(hparams, 'Train') else './eval_logs'
    os.makedirs(log_dir, exist_ok=True)

    center_initial = hparams.Data.name in ['COCO-Search18']
    tap_flag = args.tap

    run_evaluation(
        model, device, tap_flag,
        valid_img_loader_tp, valid_img_loader_ta,
        hparams_tp, hparams_ta,
        bbox_annos, human_cdf, fix_clusters,
        prior_maps_tp, prior_maps_ta,
        sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta,
        sps_test_tp, sps_test_ta,
        dataset_root, sample_action, output_saliency_metrics,
        center_initial, log_dir,
    )


if __name__ == "__main__":
    main()
