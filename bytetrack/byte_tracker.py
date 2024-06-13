import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from bytetrack.kalman_filter import KalmanFilter
from bytetrack import matching
from bytetrack.basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, ts, tlwh, score, feat=None, feat_mom=0.5, logit=None):

        # wait activate
        self._ts = ts
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.is_finalized = False

        self.score = score
        self.curr_feat = feat
        self.smooth_feat = feat
        self.feat_mom = feat_mom
        self.logit = logit
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, ts):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.history = [{
            'frame_id': frame_id,
            'score': self.score,
            'logit': self.logit,
            'feat': self.curr_feat,
            'bbox': self._tlwh,
            'ts': ts,
        }]

    def re_activate(self, new_track, frame_id, ts, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.logit = new_track.logit
        self.curr_feat = new_track.curr_feat
        self.smooth_feat = new_track.curr_feat

        self.history = [{
            'frame_id': frame_id,
            'score': new_track.score,
            'logit': new_track.logit,
            'feat': new_track.curr_feat,
            'bbox': new_track.tlwh,
            'ts': ts,
        }]

    def update(self, new_track, frame_id, ts):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.history.append({
            'frame_id': frame_id,
            'score': new_track.score,
            'logit': new_track.logit,
            'feat': new_track.curr_feat,
            'bbox': new_track.tlwh,
            'ts': ts,
        })

        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.logit = new_track.logit
        self.curr_feat = new_track.curr_feat
        if self.smooth_feat is not None and new_track.curr_feat is not None:
            self.smooth_feat = self.smooth_feat * self.feat_mom + new_track.curr_feat * (1.-self.feat_mom)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args):
        self.frame_id = -1
        self.args = args
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(args.frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.motion_weight = args.motion_weight
        assert(self.motion_weight <= 1)
        self.appearance_weight = 1 - self.motion_weight

    def reset_tracker(self):
        self.__init__(self.args)

    def get_tracks(self, min_secs=-1, min_area=-1):
        output_tracks = []
        all_tracks = self.tracked_stracks + self.lost_stracks + self.removed_stracks
        for t in all_tracks:
            ts = [h['ts'] for h in t.history]
            if min_secs>0 and ts[-1] - ts[0] < min_secs:
                continue
            area = [h['bbox'][2]*h['bbox'][3] for h in t.history]
            if min_area > 0 and np.array(area).mean() < min_area:
                continue
            output_tracks.append(t)
        return output_tracks

    def update(self, ts, boxes, scores, feats, logits):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        '''High Confidence Detections'''
        boxes_high = boxes[scores > self.args.track_thresh]
        scores_high = scores[scores > self.args.track_thresh]
        feats_high = feats[scores > self.args.track_thresh]
        logits_high = logits[scores > self.args.track_thresh]
        dets_high = [STrack(ts, STrack.tlbr_to_tlwh(tlbr), s, feat=f, logit=lgt) for
                      (tlbr, s, f, lgt) in zip(boxes_high, scores_high, feats_high, logits_high)]

        '''Low Confidence Detections'''
        inds_low = torch.logical_and(scores < self.args.track_thresh, scores > 0.1)
        boxes_low = boxes[inds_low]
        scores_low = scores[inds_low]
        logits_low = logits[inds_low]
        dets_low = [STrack(ts, STrack.tlbr_to_tlwh(tlbr), s, logit=lgt) for
                    (tlbr, s, lgt) in zip(boxes_low, scores_low, logits_low)]

        ''' Add newly detected tracklets to tracked_stracks'''
        tracked_stracks = list(filter(lambda x: x.is_activated, self.tracked_stracks))
        unconfirmed = list(filter(lambda x: not x.is_activated, self.tracked_stracks))  # type: list[STrack]

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists_boxes = matching.iou_distance(strack_pool, dets_high)
        if not self.args.mot20:
            dists_boxes = matching.fuse_score(dists_boxes, dets_high)
        dists_embs = matching.embedding_distance(strack_pool, dets_high)
        dist = (self.motion_weight*dists_boxes + self.appearance_weight*dists_embs)
        matches, u_track, u_detection = matching.linear_assignment(dist, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = dets_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, ts=ts)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, ts=ts, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, dets_low)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=self.args.track_iou_low_thresh)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = dets_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, ts=ts)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, ts=ts, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dets_high_unmatch = [dets_high[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, dets_high_unmatch)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, dets_high_unmatch)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(dets_high_unmatch[idet], self.frame_id, ts)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = dets_high_unmatch[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, ts=ts)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb