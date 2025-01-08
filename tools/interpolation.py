import numpy as np
import os
import glob

import pandas as pd

import motmetrics as mm
import sys 
from yolox.evaluators.evaluation import Evaluator


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def eval_mota(data_root, txt_path):
    accs = []
    seqs = sorted([s for s in os.listdir(data_root) if s.endswith('FRCNN')])
    for seq in seqs:
        video_out_path = os.path.join(txt_path, seq + '.txt')
        evaluator = Evaluator(data_root, seq, 'mot', anno="gt_val_half.txt")
        accs.append(evaluator.eval_file(video_out_path))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


def get_mota(data_root, txt_path):
    accs = []
    seqs = sorted([s for s in os.listdir(data_root) if s.endswith('FRCNN')])
    for seq in seqs:
        video_out_path = os.path.join(txt_path, seq + '.txt')
        evaluator = Evaluator(data_root, seq, 'mot')
        accs.append(evaluator.eval_file(video_out_path))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    mota = float(strsummary.split(' ')[-6][:-1])
    return mota


def write_results_score(filename, results):
    header = 'frame_number,track_id,class_id,score,x_topleft,y_topleft,width,height\n'
    save_format = '{frame},{id},-1,-1,{x1},{y1},{w},{h}\n'
    with open(filename, 'w') as f:
        f.write(header)  # Write the header
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[4:8]
            score = frame_data[3]
            line = save_format.format(frame=frame_id, id=track_id, x1=round(x1,1), y1=round(y1,1), w=round(w,1), h=round(h,1))
            f.write(line)


def dti(txt_path, save_path, n_min=25, n_dti=30):
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.csv')))
    for seq_txt in seq_txts:
        seq_name = seq_txt.split('\\')[-1]
        #seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',', skiprows=1)
        df = pd.read_csv(seq_txt)
        seq_data = df.to_numpy()
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 8), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 3] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 4:8]
                        left_bbox = tracklet[i - 1, 4:8]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 8), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 4:8] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 2:4] = [-1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results_score(save_seq_txt, seq_results)




if __name__ == '__main__':
    txt_path, save_path = ("C:/transmetric/dev/python/AI_camera/trial/OC_SORT-master/tracks/067-00007_Wed_Thur_27hrs_1500/2024_1204_154045_002A",
                           "C:/transmetric/dev/python/AI_camera/trial/OC_SORT-master/tracks/067-00007_Wed_Thur_27hrs_1500_interpolate/2024_1204_154045_002A_LI")

    mkdir_if_missing(save_path)
    dti(txt_path, save_path, n_min=10, n_dti=50)

























































# def dti_kitti(txt_path, save_path, n_min=30, n_dti=20):
#     seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
#     for seq_txt in seq_txts:
#         seq_name = seq_txt.split('/')[-1]
#         seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
#         min_id = int(np.min(seq_data[:, 1]))
#         max_id = int(np.max(seq_data[:, 1]))
#         seq_results = np.zeros((1, 10), dtype=np.float64)
#         for track_id in range(min_id, max_id + 1):
#             index = (seq_data[:, 1] == track_id)
#             tracklet = seq_data[index]
#             tracklet_dti = tracklet
#             if tracklet.shape[0] == 0:
#                 continue
#             n_frame = tracklet.shape[0]
#             n_conf = np.sum(tracklet[:, 6] > 0.5)
#             if n_frame > n_min:
#                 frames = tracklet[:, 0]
#                 frames_dti = {}
#                 for i in range(0, n_frame):
#                     right_frame = frames[i]
#                     if i > 0:
#                         left_frame = frames[i - 1]
#                     else:
#                         left_frame = frames[i]
#                     # disconnected track interpolation
#                     if 1 < right_frame - left_frame < n_dti:
#                         num_bi = int(right_frame - left_frame - 1)
#                         right_bbox = tracklet[i, 2:6]
#                         left_bbox = tracklet[i - 1, 2:6]
#                         for j in range(1, num_bi + 1):
#                             curr_frame = j + left_frame
#                             curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
#                                         (right_frame - left_frame) + left_bbox
#                             frames_dti[curr_frame] = curr_bbox
#                 num_dti = len(frames_dti.keys())
#                 if num_dti > 0:
#                     data_dti = np.zeros((num_dti, 10), dtype=np.float64)
#                     for n in range(num_dti):
#                         data_dti[n, 0] = list(frames_dti.keys())[n]
#                         data_dti[n, 1] = track_id
#                         data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
#                         data_dti[n, 6:] = [1, -1, -1, -1]
#                     tracklet_dti = np.vstack((tracklet, data_dti))
#             seq_results = np.vstack((seq_results, tracklet_dti))
#         save_seq_txt = os.path.join(save_path, seq_name)
#         seq_results = seq_results[1:]
#         seq_results = seq_results[seq_results[:, 0].argsort()]
#         write_results_score(save_seq_txt, seq_results)