# Validation

Desktop CI covers Linux, Windows and macOS with real Qt construction and task import. CPU SAM inference runs separately on a synthetic clip in both temporal directions.

Regression checks cover multilabel review, unknown contact, source archiving, composition changes, instance namespaces, review coverage, event filtering, correction propagation and undo. The annotation round-trip test repeats 25 save/load cycles without losing IDs, labels, evidence or null contact times.

Automated checks validate software behavior. Industrial-video track accuracy is assessed through human playback and geometry review.

The multiple-instance regression covers independent composition at the same frame, instance selection and creation through real Qt dialogs, confirmed merges and chained geometry references, review invalidation, retained historical geometry, undo/redo, 50 export/import cycles and automatic recovery. Legacy composite class aliases preserve distinct instance IDs and original coordinates. The test is included in the desktop CI matrix.

Background tracking tests run a real Qt subprocess protocol and check continued editing, progress, resident process reuse, manual edits during inference, reference reseeding, unrelated event edits, per-entity frame review, queued cancellation, retained partial results, video switching, interrupted task recovery, failed interpreter startup, undo/redo and asynchronous recovery writes under a delayed disk write. Scoped merge tests reject incomplete or mismatched candidates and preserve deleted and verified boxes.

The real SAM CPU test checks both directions, multiple targets sharing one state per direction, identical numeric IDs in the hand and object namespaces, resident model reuse, decoded-frame reuse, partial cancellation and restart. These tests run in the Linux, Windows and macOS CI matrices. Runtime depends on the annotator's hardware; synthetic inference tests do not establish accuracy or a fixed runtime for industrial clips.
