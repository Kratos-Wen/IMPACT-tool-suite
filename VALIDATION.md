# Validation

Desktop CI covers Linux, Windows and macOS with real Qt construction and task import. CPU SAM inference runs separately on a synthetic clip in both temporal directions.

Regression checks cover multilabel review, unknown contact, source archiving, composition changes, instance namespaces, review coverage, event filtering, correction propagation and undo. The annotation round-trip test repeats 25 save/load cycles without losing IDs, labels, evidence or null contact times.

Automated checks validate software behavior. Industrial-video track accuracy is assessed through human playback and geometry review.

Editor performance checks cover a 20,000-box fixture, frame-index and label-cache invalidation after edits, exact decoded pixels after preview and replay, bounded native-frame caching, coalesced hover/slider previews and precise slider release. Recovery and undo snapshots retain nested metadata without sharing live geometry. These checks run in the desktop CI matrix; timings from a compute node do not establish frame rates on annotators' computers.

Background preview checks hold a decoder behind a test barrier while Qt timers keep running, discard intermediate requests and stale results after committed seeks or video switches, recover from preview errors, and compare native pixels with an independent decoder. Preview decoding has a separate capture handle and never moves the committed annotation frame.

The multiple-instance regression covers independent composition at the same frame, instance selection and creation through real Qt dialogs, confirmed merges and chained geometry references, review invalidation, retained historical geometry, undo/redo, 50 export/import cycles and automatic recovery. Legacy composite class aliases preserve distinct instance IDs and original coordinates. The test is included in the desktop CI matrix.

Background tracking tests run a real Qt subprocess protocol and check continued editing, progress, resident process reuse, manual edits during inference, reference reseeding, unrelated event edits, per-entity frame review, queued cancellation, retained partial results, video switching, interrupted task recovery, failed interpreter startup, undo/redo and asynchronous recovery writes under a delayed disk write. Scoped merge tests reject incomplete or mismatched candidates and preserve deleted and verified boxes.

The real SAM CPU test checks both directions, multiple targets sharing one state per direction, identical numeric IDs in the hand and object namespaces, resident model reuse, decoded-frame reuse, partial cancellation and restart. These tests run in the Linux, Windows and macOS CI matrices. Runtime depends on the annotator's hardware; synthetic inference tests do not establish accuracy or a fixed runtime for industrial clips.

Keyframe workflow checks cover empty anomaly selections, multiple attributes, SOE-only geometry requirements, implicit paused-frame acceptance, excluded playback/preview visits, manual edits, saved source states and coincident event boundaries. Timeline fills retain action colors while outlines show completeness. Imported large source IDs are mapped once into video-local IDs with preserved aliases; reopen and deletion do not renumber remaining instances. macOS checks cover readable font sizes and scale changes.

BBox navigation checks cover role-specific frame ticks, drawing/viewing/model provenance, duplicate warnings, exact decoded-frame clicks, cache invalidation after edits and undo, selective machine import, and preserved object references. A 20,000-box fixture checks 200 repeated marker queries without rescanning tracks and 100 exact jumps without changing IDs. Human evidence survives every import mode; conflicting machine hands are omitted in hands-only mode while the source draft is retained.
