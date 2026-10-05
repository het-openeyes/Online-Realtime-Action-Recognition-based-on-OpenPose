# -*- coding: UTF-8 -*-
"""
Headless version of Action/recognizer.py for the HTTP API.

Same pipeline as main.py (OpenPose pose estimation -> DeepSort tracking -> framewise DNN action
classifier), but it returns data instead of drawing on the frame, and keeps one tracker per camera so
several camera feeds can be analysed by the same process.
"""
import time
from pathlib import Path
from collections import Counter, deque

import cv2 as cv
import numpy as np

from Action.action_enum import Actions
from Action.recognizer import load_action_premodel
from Pose.coco_format import CocoPart, CocoPairsRender
from Tracking import generate_dets as gdet
from Tracking.deep_sort import preprocessing
from Tracking.deep_sort.detection import Detection
from Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Tracking.deep_sort.tracker import Tracker
from Pose.pose_visualizer import TfPoseVisualizer

POSE_INPUT = {'VGG_origin': (656, 368), 'mobilenet_thin': (432, 368)}  # network input size per pose model

N_JOINTS = CocoPart.Background.value  # 18 COCO joints, 36 features per person
MIN_JOINTS = 5                         # fewer than this is not enough body to classify

# (rotation applied to the frame, mapping of a joint found in the rotated frame back to the original)
ROTATIONS = [
    (cv.ROTATE_90_CLOCKWISE, lambda u, v: (v, 1.0 - u)),
    (cv.ROTATE_90_COUNTERCLOCKWISE, lambda u, v: (1.0 - v, u)),
]


# zoomed fallback: 3 x 2 overlapping crops, each 62% x 67% of the frame
CROP_SIZE = (0.62, 0.67)
CROP_ORIGINS = [(x, y) for y in (0.0, 0.33) for x in (0.0, 0.19, 0.38)]


def _box(s):
    xs, ys = [p[0] for p in s.values()], [p[1] for p in s.values()]
    return min(xs), min(ys), max(xs), max(ys)


def overlap(a, b):
    """Intersection over the smaller of two skeletons' joint boxes."""
    ax0, ay0, ax1, ay1 = _box(a)
    bx0, by0, bx1, by1 = _box(b)
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)) or 1e-6
    return iw * ih / small


def box_iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / union if union > 0 else 0.0


def body_tilt(joints, w, h):
    """Angle of the torso (neck to mid-hip) from vertical, in degrees; None without neck and a hip."""
    hips = [joints[i] for i in (8, 11) if i in joints]
    if 1 not in joints or not hips:
        return None
    hx, hy = sum(p[0] for p in hips) / len(hips), sum(p[1] for p in hips) / len(hips)
    dx, dy = (hx - joints[1][0]) * w, (hy - joints[1][1]) * h
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


class CameraState:
    """Tracker and label history for one camera."""

    def __init__(self):
        self.tracker = Tracker(NearestNeighborDistanceMetric('cosine', 0.3, None))
        self.labels = {}
        self.centers = {}
        self.down = None  # {'since', 'bbox'} while a fall is unresolved
        self.last_seen = time.time()


class PoseActionService:
    def __init__(self, pose_model='mobilenet_thin', action_model='Action/framewise_recognition_under_scene.h5',
                 appearance_model='Tracking/graph_model/mars-small128.pb', smooth_len=3, motion_len=4,
                 walk_move_ratio=0.08, fall_tilt=60, down_hold=60, min_extent=0.1):
        # VGG_origin is slower but finds small and lying people far more often; it was trained at 656x368
        self.estimator = TfPoseVisualizer(str(Path.cwd() / 'Pose/graph_models' / pose_model / 'graph_opt.pb'),
                                          target_size=POSE_INPUT[pose_model])
        self.pose_model = pose_model
        self.classifier = load_action_premodel(action_model)
        self.encoder = gdet.create_box_encoder(appearance_model, batch_size=1)
        self.smooth_len, self.motion_len, self.walk_move_ratio = smooth_len, motion_len, walk_move_ratio
        self.fall_tilt, self.down_hold, self.min_extent = fall_tilt, down_hold, min_extent
        self.cameras = {}
        self.actions = [a.name for a in Actions]

    def reset(self, camera):
        self.cameras.pop(camera, None)

    def _state(self, camera):
        st = self.cameras.get(camera)
        if st is None or time.time() - st.last_seen > 30:  # stale feed: start tracking afresh
            st = self.cameras[camera] = CameraState()
        st.last_seen = time.time()
        return st

    def _detect(self, frame):
        """OpenPose on one frame: a list of {joint index: (x, y)} in 0..1 frame fractions."""
        out = []
        for human in self.estimator.inference(frame):
            joints = {i: (bp.x, bp.y) for i, bp in human.body_parts.items() if i < N_JOINTS}
            if len(joints) >= MIN_JOINTS:
                out.append(joints)
        return out

    def _detect_zoomed(self, frame):
        """Fallback when the whole frame finds nobody.

        OpenPose shrinks every frame to its 432x368 input, so a person far from a ceiling camera is only a
        few dozen pixels tall, and it was trained on upright people, so it misses someone lying down.
        Overlapping crops make the person bigger; turning each crop 90 degrees makes a lying body look
        upright. Joints are mapped back to whole-frame fractions and duplicates from overlapping crops
        are dropped.
        """
        h, w = frame.shape[:2]
        found = []
        for cx, cy in CROP_ORIGINS:
            x0, y0 = int(cx * w), int(cy * h)
            cw, ch = int(CROP_SIZE[0] * w), int(CROP_SIZE[1] * h)
            crop = frame[y0:y0 + ch, x0:x0 + cw]
            to_frame = lambda u, v: ((x0 + u * cw) / w, (y0 + v * ch) / h)
            variants = [(crop, lambda u, v: (u, v))] + [(cv.rotate(crop, rot), back) for rot, back in ROTATIONS]
            for img, back in variants:
                hits = [{i: to_frame(*back(*xy)) for i, xy in s.items()} for s in self._detect(img)]
                if hits:
                    found += hits
                    break
        # keep the most complete skeleton among those that overlap
        found.sort(key=len, reverse=True)
        kept = []
        for s in found:
            if all(overlap(s, k) < 0.3 for k in kept):
                kept.append(s)
        return kept

    def analyze(self, frame, camera='default'):
        """frame: BGR image. Returns a JSON-able dict; coordinates are 0..1 fractions of the frame."""
        t0 = time.time()
        h, w = frame.shape[:2]
        st = self._state(camera)
        skeletons, rotated = self._detect(frame), False
        if not skeletons:
            skeletons, rotated = self._detect_zoomed(frame), True

        people = []
        for joints in skeletons:
            feats = []
            for i in range(N_JOINTS):  # missing joints are zero-filled, as in training
                feats += [round(joints[i][0], 2), round(joints[i][1], 2)] if i in joints else [0.0, 0.0]
            xs = [int(x * w + 0.5) for x, _ in joints.values()]
            ys = [int(y * h + 0.5) for _, y in joints.values()]
            if max(max(xs) - min(xs), max(ys) - min(ys)) < self.min_extent * h:
                continue  # a few joints bunched in a corner of the frame: furniture, not a person
            probs = self.classifier.predict(np.array(feats, dtype=float).reshape(1, 36))[0]
            people.append({'joints': {i: [round(x, 4), round(y, 4)] for i, (x, y) in joints.items()}, 'probs': probs,
                           'bbox': [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                           'tilt': body_tilt(joints, w, h)})

        # track identities across frames
        ids = [None] * len(people)
        if people:
            boxes = np.array([p['bbox'] for p in people], dtype=float)
            feats = self.encoder(frame, boxes)
            dets = [Detection(b, 1.0, f) for b, f in zip(boxes, feats)]
            keep = preprocessing.non_max_suppression(np.array([d.tlwh for d in dets]), 1.0, np.ones(len(dets)))
            st.tracker.predict(); st.tracker.update([dets[i] for i in keep])
            for trk in st.tracker.tracks:
                if not trk.is_confirmed() or trk.time_since_update > 1:
                    continue
                tx, ty, tw, th = trk.to_tlwh()
                d = [np.hypot((p['bbox'][0] + p['bbox'][2] / 2) - (tx + tw / 2), (p['bbox'][1] + p['bbox'][3] / 2) - (ty + th / 2)) for p in people]
                j = int(np.argmin(d))
                if ids[j] is None:
                    ids[j] = int(trk.track_id)

        out = []
        for p, tid in zip(people, ids):
            label = self.actions[int(np.argmax(p['probs']))]
            conf = float(np.max(p['probs']))
            source = 'model'
            # The framewise model only knows the falls in its training scene. A torso lying closer to
            # horizontal than vertical is a fall whatever it predicts.
            if label != 'fall_down' and p['tilt'] is not None and p['tilt'] > self.fall_tilt:
                label, source = 'fall_down', 'body-angle'
            cx = p['bbox'][0] + p['bbox'][2] / 2.0
            if tid is not None:  # majority vote over recent frames; a moving box upgrades stand to walk
                hist = st.labels.setdefault(tid, deque(maxlen=self.smooth_len)); hist.append(label)
                if label != 'fall_down':  # a fall is never voted away: it may only be visible for one frame
                    label = Counter(hist).most_common(1)[0][0]
                cs = st.centers.setdefault(tid, deque(maxlen=self.motion_len)); cs.append(cx)
                if label == 'stand' and len(cs) == self.motion_len and max(cs) - min(cs) > self.walk_move_ratio * w:
                    label = 'walk'
            bx, by, bw, bh = p['bbox']
            out.append({'id': tid, 'action': label, 'confidence': round(conf, 3), 'source': source,
                        'tilt': None if p['tilt'] is None else round(p['tilt']), 'rotated': rotated,
                        'probabilities': {a: round(float(v), 3) for a, v in zip(self.actions, p['probs'])},
                        'bbox': [round(bx / w, 4), round(by / h, 4), round(bw / w, 4), round(bh / h, 4)],
                        'joints': p['joints']})
        # Person down: once a fall is seen the alert holds while nobody is seen upright at that spot. Someone
        # lying still is exactly what the pose model loses, so losing them must not clear the alarm.
        now = time.time()
        falls = [o for o in out if o['action'] == 'fall_down']
        if falls:
            st.down = {'since': st.down['since'] if st.down else now, 'bbox': falls[0]['bbox']}
        elif st.down:
            up = [o for o in out if o['action'] in ('stand', 'walk') and box_iou(o['bbox'], st.down['bbox']) > 0.1]
            if up or now - st.down['since'] > self.down_hold:
                st.down = None
        alerts = ['fall_down'] if st.down else []
        down = None if not st.down else {'bbox': st.down['bbox'], 'seconds': round(now - st.down['since'], 1),
                                         'seen_now': bool(falls)}
        return {'camera': camera, 'width': w, 'height': h, 'people': out, 'alerts': alerts, 'person_down': down,
                'skeleton': [list(p) for p in CocoPairsRender], 'ms': round((time.time() - t0) * 1000)}
