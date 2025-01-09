import argparse
import os
import os.path as osp
import time
import cv2
import numpy as np
import torch
import sys
from loguru import logger
import csv

from sympy import false

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

#
# def get_base_directory():
#     """
#     Get the base directory where the application is running.
#     """
#     if getattr(sys, 'frozen', False):  # Check if running as a bundled executable
#         # Always set the base directory to the persistent storage location
#         home_dir = os.path.expanduser("~")
#         base_dir = os.path.join(home_dir, "COUNT_FILES", "tupi-ai-realtime")
#     else:
#         base_dir = os.path.dirname(os.path.abspath(__file__)) # Use script's original directory
#
#     return base_dir


def create_track_file(output_dir, folder_path, track_id):

    dir_name = folder_path.split("/")[-3]

    full_path = os.path.join(output_dir, dir_name, folder_path.split('/')[-2])

    # Ensure the directory exists
    os.makedirs(full_path, exist_ok=True)

    filename = f"{track_id}.csv"

    filepath = os.path.join(full_path, filename)

    return filepath



def read_detections_from_csv_folder(folder_path):
    """
    Reads detections from multiple CSV files in the specified folder.
    Each CSV file corresponds to detections for a single frame.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        dict: A dictionary where keys are frame numbers (int) derived from CSV filenames,
        and values are lists of detections (xmin, ymin, xmax, ymax, score, label).
    """
    detection_dict = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            try:
                # Extract frame number from the file name (assumes frame_<frame_number>.csv format)
                frame_number = int(file_name.split('.')[0])
                file_path = os.path.join(folder_path, file_name)

                # Initialize the list of detections for this frame
                detection_dict[frame_number] = []

                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    next(csv_reader, None)  # Skip the first row (header)

                    for row in csv_reader:
                        try:
                            # Each row is expected to contain: xmin, ymin, xmax, ymax, score, label
                            # time = float(row[0])
                            xmin = float(row[0])
                            ymin = float(row[1])
                            xmax = float(row[2])
                            ymax = float(row[3])
                            score = float(row[4])
                            label = int(row[5])
                            detection_dict[frame_number].append([xmin, ymin, xmax, ymax, score, label])
                        except (ValueError, IndexError):
                            print(f"Skipping invalid row in file {file_name}: {row}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return detection_dict, folder_path

def run_track(args):

    # detection_data = args.detection_data.replace("\\", "/")
    # output = args.output.replace("\\", "/")

    detection_data = args.detection_data
    output = args.output

    tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
    results = []

    # Read detections from the specified folder
    detections, current_folder = read_detections_from_csv_folder(detection_data)

    # Process each frame (frame numbers may be non-continuous)
    for frame_number in sorted(detections.keys()):
        outputs_by_frame = detections[frame_number]

        if len(outputs_by_frame) > 0:

            detections_tensor = [torch.tensor(outputs_by_frame, dtype=torch.float32)]

            if detections_tensor[0] is not None:
                online_targets = tracker.update(detections_tensor[0], 480, 640)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] < tlwh[3]
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_number},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )


                        # Save tracking data into per-track CSV files
                        track_file = create_track_file(output, current_folder, int(tid))

                        with open(track_file, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            if csvfile.tell() == 0:
                                csv_writer.writerow(
                                    ["frame_number", "track_id", "class_id", "score", "x_topleft", "y_topleft", "width", "height"])
                            csv_writer.writerow([frame_number, int(tid), 0, 0, round(tlwh[0], 1), round(tlwh[1], 1),
                                                 round(tlwh[2], 1), round(tlwh[3], 1)])

        #
        #
        # if args.save_result:
        #     res_file = osp.join(vis_folder, f"{current_folder.split('/')[-2]}.txt")
        #     with open(res_file, 'w') as f:
        #         f.writelines(results)
        #     logger.info(f"save results to {res_file}")


# def main(exp, args):
#     if not args.expn:
#         args.expn = exp.exp_name
#
#     output_dir = osp.join(exp.output_dir, args.expn)
#     os.makedirs(output_dir, exist_ok=True)
#
#     args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
#
#     logger.info("Args: {}".format(args))
#
#     model = exp.get_model().to(args.device)
#     logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
#     model.eval()
#
#     # current_time = time.localtime()
#
#     run_track(args)


if __name__ == "__main__":
    args = make_parser("config.yaml").parse_args()
    # exp = get_exp(args.exp_file, args.name)
    # main(exp, args)
    run_track(args)
