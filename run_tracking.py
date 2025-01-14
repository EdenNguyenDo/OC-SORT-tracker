
import os
import torch
import csv

from torchvision.ops import nms
from trackers.ocsort_tracker.ocsort import OCSort

from utils.args import make_parser



def create_track_file(output_dir, folder_path, track_id):

    full_path = os.path.join(output_dir, folder_path.split('/')[-2])

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

    return detection_dict


def apply_nms(detections_tensor, iou_threshold):

    # Extract bounding boxes, scores, and labels
    boxes = detections_tensor[0][:, :4]  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    scores = detections_tensor[0][:, 4]  # Confidence scores
    labels = detections_tensor[0][:, 5]  # Class labels

    # Apply Non-Maximum Suppression (NMS)
    keep_inds = nms(boxes, scores, iou_threshold)

    # Keep the detections based on NMS filtering
    kept_boxes = boxes[keep_inds]
    kept_scores = scores[keep_inds]
    kept_labels = labels[keep_inds]

    # Filter the detections that passed NMS
    kept_detections = torch.cat((kept_boxes, kept_scores.unsqueeze(1), kept_labels.unsqueeze(1)), dim=1)
    detections_nms = [kept_detections]

    return detections_nms



def run_track(args):

    # detection_data = args.detection_data.replace("\\", "/")
    # output = args.output.replace("\\", "/")

    detection_data_filepath = args.detection_data
    output = args.output

    tracker = OCSort(det_thresh=args.track_thresh, lower_det_thresh=args.lower_track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte,
                     inertia=args.inertia, min_hits=args.min_hits, max_age=args.track_buffer, asso_func=args.asso, delta_t=args.deltat)
    results = []

    line_count_dict = {}

    # Read detections from the specified folder
    detections = read_detections_from_csv_folder(detection_data_filepath)

    sorted_keys = sorted(detections.keys())

    # Process each frame (frame numbers may be non-continuous)
    for frame_number in sorted_keys:
        outputs_by_frame = detections[frame_number]

        if len(outputs_by_frame) > 0:

            detections_tensor = [torch.tensor(outputs_by_frame, dtype=torch.float32)]

            # Apply NMS to remove overlapping boxes.
            detections_nms = apply_nms(detections_tensor, args.nms_iou_thresh)

            if detections_tensor[0] is not None:
                online_targets = tracker.update(detections_nms[0], 480, 640)
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

                        # Update line count for the current tid
                        if tid not in line_count_dict:
                            line_count_dict[tid] = 0

                        # Only proceed if the line count is below the threshold
                        if line_count_dict[tid] <= args.max_exist:
                            track_file = create_track_file(output, detection_data_filepath, int(tid))

                            with open(track_file, 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                if csvfile.tell() == 0:
                                    csv_writer.writerow(
                                        ["frame_number", "track_id", "class_id", "score", "x_topleft", "y_topleft",
                                         "width", "height"])

                                csv_writer.writerow([
                                    frame_number, int(tid), 0, 0,
                                    round(tlwh[0], 1), round(tlwh[1], 1),
                                    round(tlwh[2], 1), round(tlwh[3], 1)
                                ])

                                # Increment line count for tid
                                line_count_dict[tid] += 1
                        # else:
                        #     print(f"Skipping vehicle track {tid}: it retains for over {args.max_exist} frames")


if __name__ == "__main__":
    args = make_parser("config.yaml").parse_args()
    run_track(args)
