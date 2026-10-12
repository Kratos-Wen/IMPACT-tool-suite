import ast
from pathlib import Path
from types import SimpleNamespace


# Exercise navigation without requiring Qt or a loaded video.
source = Path(__file__).resolve().parents[1] / 'ui' / 'frame_review.py'
tree = ast.parse(source.read_text())
mixin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FrameReviewMixin')
method = next(node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == '_jump_frame_review')
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)

issues = [
    dict(event_id=1, hand='Left_hand', frame=10),
    dict(event_id=2, hand='Right_hand', frame=30),
]

for direction, current, target in [(1, 20, issues[1]), (-1, 20, issues[0]), (1, 40, issues[0]), (-1, 0, issues[1])]:
    calls = []
    player = SimpleNamespace(current_frame=current)

    def seek(frame):
        player.current_frame = frame
        calls.append(('seek', frame))

    def refresh(frame):
        assert player.current_frame == frame
        calls.append(('boxes', frame))

    player.seek = seek
    window = SimpleNamespace(
        player=player,
        _frame_review_issues=lambda: issues,
        _pause=lambda: calls.append(('pause',)),
        _set_selected_event=lambda event, hand: calls.append(('select', event, hand)),
        _refresh_boxes_for_frame=refresh,
        _set_frame_controls=lambda frame: calls.append(('controls', frame)),
    )
    namespace['_jump_frame_review'](window, direction)
    assert calls == [
        ('pause',),
        ('select', target['event_id'], target['hand']),
        ('seek', target['frame']),
        ('boxes', target['frame']),
        ('controls', target['frame']),
    ]

window._frame_review_issues = lambda: []
calls.clear()
namespace['_jump_frame_review'](window, 1)
assert calls == []
print('FRAME_REVIEW_NAVIGATION_REFRESH_PASS')
