# Targeted reviewer notes

Put the supplied `review_flags.json` next to the original task `video.mp4`, then reopen the video. The Review Summary shows the pending count. **Next reviewer note** selects the referenced event and jumps to its source frame. Orange lines appear inside the selected event; no extra timeline rows are added.

After correcting a flagged item, **Note addressed** dismisses that reminder. The status is saved atomically in `review_flags.resolved.json`; it does not approve an event or modify any annotation. Saving incomplete drafts remains allowed. Source task name and frame count must match. Notes are for the returned video version; do not reuse them with a differently trimmed video.

The note file is read once per opened video. A replacement note file takes effect after reopening. Existing resolutions are bound to the exact note file, so an updated batch cannot inherit stale dismissals.

Audio tasks can provide a `review_hint` on an utterance. It is highlighted above the transcript and reachable with **Next reviewer note**. It never checks text or timing automatically.
