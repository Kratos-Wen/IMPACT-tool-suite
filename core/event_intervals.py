"""Closed event intervals: every boundary frame belongs to its event."""
def interiors_overlap(start,end,other_start,other_end):
    """Compatibility name; sharing even one frame is an overlap."""
    return not (end < other_start or start > other_end)
