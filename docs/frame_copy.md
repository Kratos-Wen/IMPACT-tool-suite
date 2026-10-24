# Copy boxes between frames

1. Select the source event and a frame with the boxes you need.
2. Press **Ctrl/⌘+Shift+C**.
3. Jump to the target frame, select its event and press **Ctrl/⌘+Shift+V**.
4. Drag or resize the pasted boxes, then save as usual.

The same actions are in **File → Keyframes**. Only hands, Object and Instrument participating in the event are copied. Instance IDs stay unchanged. Across events, an entity is pasted only if that same ID participates in the target event. Existing target boxes are skipped; remove an incorrect target box first when replacing it.

Copying does not change the source frame or extend a track across intermediate frames. Pasted boxes are editable, undoable and included in autosave. Their source frame is retained in the saved annotation. The clipboard is limited to the current video.

Saved `reviewed*.json` files retain their existing object IDs. An explicitly cleared Object or Instrument remains empty even if an older serialized link is still present.
