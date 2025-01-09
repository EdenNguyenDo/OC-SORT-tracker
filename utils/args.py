import argparse
import yaml

def make_parser(config_path=None):
    # Initialize config to None by default
    config = None
    if config_path is not None:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

    # Fallback function to safely get values from config or return a default
    def get_config_value(key, default=None):
        if config is not None:
            return config.get(key, default)
        return default

    parser = argparse.ArgumentParser("OC-SORT parameters")

    # General Parameters
    parser.add_argument("--expn", type=str, default=get_config_value('expn'), help="Experiment name")
    parser.add_argument("-n", "--name", type=str, default=get_config_value('name'), help="Model name")

    # Distributed Training
    parser.add_argument("--dist-backend", default=get_config_value('dist-backend', "nccl"), type=str, help="Distributed backend")
    parser.add_argument("--output_dir", type=str, default=get_config_value('output_dir', "evaldata/trackers/mot_challenge"), help="Output directory")
    parser.add_argument("--dist-url", default=get_config_value('dist-url'), type=str, help="URL used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=get_config_value('batch-size', 64), help="Batch size")
    parser.add_argument("-d", "--devices", default=get_config_value('devices', 'cpu'), type=str, help="Device for training")

    # Distributed Training Specific
    parser.add_argument("--local_rank", default=get_config_value('local_rank', 0), type=int, help="Local rank for distributed training")
    parser.add_argument("--num_machines", default=get_config_value('num_machines', 1), type=int, help="Number of nodes for training")
    parser.add_argument("--machine_rank", default=get_config_value('machine_rank', 0), type=int, help="Node rank for multi-node training")

    # Experiment File
    parser.add_argument("-f", "--exp_file", default=get_config_value('exp_file'), type=str, help="Experiment description file")

    # Precision and Optimization Settings
    parser.add_argument("--fp32", dest="fp32", default=get_config_value('fp32', False), action="store_true", help="Adopting mixed precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=get_config_value('fuse', False), action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=get_config_value('trt', False), action="store_true", help="Use TensorRT model for testing.")
    parser.add_argument("--test", dest="test", default=get_config_value('test', False), action="store_true", help="Evaluating on test-dev set.")
    parser.add_argument("--speed", dest="speed", default=get_config_value('speed', False), action="store_true", help="Speed test only.")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # Detection Parameters
    parser.add_argument("-c", "--ckpt", default=get_config_value('ckpt'), type=str, help="Checkpoint for evaluation")
    parser.add_argument("--conf", default=get_config_value('conf', 0.12), type=float, help="Test confidence threshold")
    parser.add_argument("--nms", default=get_config_value('nms', 0.1), type=float, help="Non-maximum suppression threshold")
    parser.add_argument("--tsize", default=get_config_value('tsize'), type=int, help="Test image size")
    parser.add_argument("--seed", default=get_config_value('seed'), type=int, help="Evaluation seed")

    # Tracking Parameters
    parser.add_argument("--track_thresh", type=float, default=get_config_value('track_thresh', 0.8), help="Detection confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=get_config_value('iou_thresh', 0.2), help="IOU threshold for SORT matching")
    parser.add_argument("--min_hits", type=int, default=get_config_value('min_hits', 6), help="Minimum hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=get_config_value('inertia', 0.4), help="Weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=get_config_value('deltat', 1.5), help="Time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=get_config_value('track_buffer', 60), help="Frames for keeping lost tracks")
    parser.add_argument("--match_thresh", type=float, default=get_config_value('match_thresh', 0.85), help="Matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=get_config_value('min-box-area', 150), help='Filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default=get_config_value('gt-type', "_val_half"), help="Suffix to find the GT annotation")
    parser.add_argument("--mot20", dest="mot20", default=get_config_value('mot20', False), action="store_true", help="Test MOT20.")
    parser.add_argument("--public", action="store_true", help="Use public detection")
    parser.add_argument('--asso', default=get_config_value('asso', "giou"), help="Similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=get_config_value('use_byte', True), action="store_true", help="Use byte in tracking.")

    # KITTI/BDD100K Inference with Public Detections
    parser.add_argument('--raw_results_path', type=str, default=get_config_value('raw_results_path', "exps/permatrack_kitti_test/"), help="Path to raw tracking results")
    parser.add_argument('--out_path', type=str, default=get_config_value('out_path'), help="Path to save output results")
    parser.add_argument("--dataset", type=str, default=get_config_value('dataset', "mot"), help="Dataset type (kitti or bdd)")
    parser.add_argument("--hp", action="store_true", help="Use head padding to add missing objects during initializing the tracks (offline).")

    # Demo Video Settings
    parser.add_argument("--demo_type", default=get_config_value('demo_type', "video"), help="Demo type (image, video, or webcam)")
    parser.add_argument("--path", default=get_config_value('path', "./videos/dance_demo.mp4"), help="Path to images or video")
    parser.add_argument("--camid", type=int, default=get_config_value('camid', 0), help="Webcam demo camera ID")
    parser.add_argument("--save_result", action="store_true", help="Whether to save the inference result of image/video")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=get_config_value('aspect_ratio_thresh', 1.5), help="Threshold for filtering out boxes of which aspect ratio is above the given value.")
    parser.add_argument('--min_box_area', type=float, default=get_config_value('min_box_area', 8), help='Filter out tiny boxes')
    parser.add_argument("--device", default=get_config_value('device', "gpu"), type=str, help="Device to run the model (cpu or gpu)")

    # Detection File Path
    parser.add_argument('--detection_data', default=config.get("detection_data"), type=str, help="Path to the raw detection file")

    return parser
