# Validation mode

Use the existing **Validate** switch. Enter your reviewer name once per session. Editing, drawing boxes, SAM and saving work as before.

After reviewing the whole video, click **Mark validated**, then **Ctrl+S**. The annotation JSON stores the reviewer, UTC timestamp and video-level validation status. The top bar shows **Validated ✓**, including after reopening the saved annotation.

Changing events, boxes, object identities or assembly/wire state changes the status to **Needs revalidation**. Undoing back to the reviewed content restores its previous status. Playback, frame navigation and switching modes do not invalidate it. Recovery drafts include the status.

Entering Validation mode alone does not mark a video validated. Incomplete drafts can still be saved through the existing warning. No per-frame or per-event approval clicks are required.
