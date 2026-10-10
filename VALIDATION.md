# Validation

Desktop CI covers Linux, Windows and macOS with real Qt construction and task import. CPU SAM inference runs separately on a synthetic clip in both temporal directions.

Regression checks cover multilabel review, unknown contact, source archiving, composition changes, instance namespaces, review coverage, event filtering, correction propagation and undo. The annotation round-trip test repeats 25 save/load cycles without losing IDs, labels, evidence or null contact times.

Automated checks validate software behavior. Industrial-video track accuracy is assessed through human playback and geometry review.

The multiple-instance regression covers independent composition at the same frame, instance selection and creation through real Qt dialogs, confirmed merges and chained geometry references, review invalidation, retained historical geometry, undo/redo, 50 export/import cycles and automatic recovery. Legacy composite class aliases preserve distinct instance IDs and original coordinates. The test is included in the desktop CI matrix.
