# -*- coding: UTF-8 -*-
"""
HTTP API around the pose/action pipeline, used by the opened-eyes viewer's CCTV feeds.

  GET  /health                     -> {"ok": true, "actions": [...]}
  POST /analyze?camera=C8          body: a JPEG or PNG frame  -> people, joints, actions, alerts
  POST /reset?camera=C8            forget the tracker for one camera

Requests are handled one at a time on the main thread: the TensorFlow 1.x graph and Keras session are
not safe to share across threads.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(description='Pose/action recognition API')
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=5200)
parser.add_argument('--origin', default='*', help='Access-Control-Allow-Origin value')
parser.add_argument('--pose-model', default='VGG_origin', choices=['VGG_origin', 'mobilenet_thin'],
                    help='VGG_origin is slower but more accurate (download it with Pose/graph_models/VGG_origin/download.sh)')
args = parser.parse_args()

from Action.api_service import PoseActionService  # noqa: E402  (heavy import, after arg parsing)

service = PoseActionService(pose_model=args.pose_model)
MAX_BODY = 8 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', args.origin)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', args.origin)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path.rstrip('/').endswith('/health'):
            return self._send(200, {'ok': True, 'actions': service.actions})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        url = urlparse(self.path)
        camera = parse_qs(url.query).get('camera', ['default'])[0][:32]
        route = url.path.rstrip('/').rsplit('/', 1)[-1]
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY:
            return self._send(413, {'error': 'frame too large'})
        body = self.rfile.read(length)
        if route == 'reset':
            service.reset(camera)
            return self._send(200, {'ok': True})
        if route != 'analyze':
            return self._send(404, {'error': 'not found'})
        frame = cv.imdecode(np.frombuffer(body, np.uint8), cv.IMREAD_COLOR)
        if frame is None:
            return self._send(400, {'error': 'body is not a decodable image'})
        try:
            self._send(200, service.analyze(frame, camera))
        except Exception as e:  # keep serving: one bad frame should not end the process
            self._send(500, {'error': str(e)})

    def log_message(self, fmt, *a):
        pass


if __name__ == '__main__':
    print('Pose/action API on http://%s:%d' % (args.host, args.port), flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()
