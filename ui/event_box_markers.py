"""Cached bbox ticks inside an existing event rectangle; no additional widget."""
from bisect import bisect_left, bisect_right

from core.bbox_markers import CAPTIONS


class EventBoxMarkers:
    def __init__(self, row, get_markers):
        self.row = row
        self.get_markers = get_markers
        self._key = None
        self.groups = {}
        self.pixels = []
        self.rebuilds = 0

    def layout(self, segment):
        row = self.row
        markers = self.get_markers() or ()
        key = (id(markers), segment['event_id'], segment['start'], segment['end'],
               row.get_vs(), row.get_span(), row.width(), row.get_gutter())
        if key == self._key:
            return self.groups
        groups = {}
        start = max(segment['start'], row.get_vs())
        end = min(segment['end'], row.get_vs() + row.get_span())
        for marker in markers:
            frame = marker['frame']
            if not start <= frame <= end:
                continue
            x = row.frame_to_x(frame)
            if not row.get_gutter() <= x < row.width():
                continue
            group = groups.setdefault(x, {'rows': {}, 'conflict': False})
            group['rows'].setdefault(frame, []).append(marker)
            group['conflict'] |= marker['state'] in ('conflict', 'reviewer')
        for group in groups.values():
            group['frames'] = sorted(group['rows'])
        self.groups = groups
        self.pixels = sorted(groups)
        self._key = key
        self._source = markers  # Retain the list while its identity is cached.
        self.rebuilds += 1
        return groups

    def hit(self, segment, x, y):
        row = self.row
        # The top of the event remains available for SOE handle dragging.
        if x < row.get_gutter() or not 18 <= y <= row.height() - 9:
            return None
        self.layout(segment)
        pixels = self.pixels[bisect_left(self.pixels, x - 3):bisect_right(self.pixels, x + 3)]
        if not pixels:
            return None
        pixel = min(pixels, key=lambda value: (abs(value - x), value))
        group = self.groups[pixel]
        target = row.x_to_frame(x)
        frames = group['frames']
        index = bisect_left(frames, target)
        candidates = frames[max(0, index - 1):index + 1]
        frame = min(candidates, key=lambda value: (abs(value - target), value))
        details = ' · '.join(dict.fromkeys(
            f"{marker['role']} {marker['entity']}: {CAPTIONS[marker['state']]}"
            for marker in group['rows'][frame]))
        return {'frame': frame, 'tooltip': f'Frame {frame} · {details}'}
