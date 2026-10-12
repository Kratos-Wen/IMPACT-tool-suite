# Offline audio review

Open the supplied `review.html` in a desktop browser on Windows, macOS or Linux. Keep its accompanying `audio` folder beside it. No server, model or Python installation is needed to review a prepared package.

Each screen contains one utterance, its player, editable transcript and prefilled start/end in original audio seconds. Playback stops at the selected utterance's end. The original ASR clip and a separate two-second context clip are available when checking boundaries.

Tick **Text correct** and **Timing correct**, or use **Approve & next**. Changing text clears its confirmation; changing boundaries clears timing confirmation. **Exclude** removes an utterance from the final transcript while keeping its review record.

Edits autosave in the same browser. **Save review file** exports approved entries in `items` and all working records in `review_records`; **Import review** resumes from this file. Previous review exports are supported. Task and source-text checks prevent importing another recording's edits. A visible message reports storage failures; exporting still uses the current in-memory edits.

Shortcuts: Space plays/pauses outside text controls, Alt+Left/Right changes utterance, and Ctrl/Command+Enter approves and advances.

To generate a self-contained page from your own manifest:

```bash
python audio_review/render.py manifest.json review.html --ui-language en
```

The manifest contains `language`, `plan_hash`, `files` and `items`. Each file supplies `source_relative_path`, `source_sha256`, `duration_seconds` and `audio_file`. Each utterance supplies `review_id`, `utterance_id`, `source_relative_path`, `source_text_sha256`, `original_text`, `original_start_audio_s`, `original_end_audio_s` and `audio_file`. Optional `initial_reviews` is indexed by `review_id`. Interface languages are `en`, `zh` and `de`.

Run real-browser checks with `python -m pip install selenium`, then `python -m unittest discover -s tests -p test_audio_review_browser.py -v`. Firefox is the default; set `IMPACT_AUDIO_BROWSER=chrome` for Chrome. The tests generate synthetic audio and transcripts.
