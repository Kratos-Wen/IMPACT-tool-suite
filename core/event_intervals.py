"""Inclusive keyframe boundaries permit adjacent events to share a transition."""
def interiors_overlap(start,end,other_start,other_end):
    return (start==end==other_start==other_end or
            not (end<=other_start or start>=other_end))
