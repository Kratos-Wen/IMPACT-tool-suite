/* Review state is independent of the player. Original ASR records stay immutable. */
const AudioReviewCore = (() => {
  const number = value => value === '' || value == null ? NaN : Number(value);
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  function validTimes(review, item, file) {
    const start = number(review.human_start_audio_s), end = number(review.human_end_audio_s);
    const duration = number(file?.duration_seconds);
    return Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end > start
      && (!Number.isFinite(duration) || end <= duration + 0.001);
  }
  function normalize(item, previous = {}, file = {}) {
    const corrected = previous.timing_decision === 'corrected';
    const currentFormat = own(previous, 'timing_checked');
    const explicitTimes = Number.isFinite(number(previous.human_start_audio_s))
      && Number.isFinite(number(previous.human_end_audio_s));
    const review = {
      human_text: previous.human_text ?? item.original_text ?? '',
      human_start_audio_s: currentFormat ? (previous.human_start_audio_s ?? '') : Number.isFinite(number(previous.human_start_audio_s))
        ? number(previous.human_start_audio_s) : number(item.original_start_audio_s),
      human_end_audio_s: currentFormat ? (previous.human_end_audio_s ?? '') : Number.isFinite(number(previous.human_end_audio_s))
        ? number(previous.human_end_audio_s) : number(item.original_end_audio_s),
      text_checked: own(previous, 'text_checked') ? previous.text_checked === true
        : ['keep_as_spoken', 'correct_asr'].includes(previous.text_decision),
      timing_checked: own(previous, 'timing_checked') ? previous.timing_checked === true
        : previous.timing_decision === 'keep_original' || (corrected && explicitTimes),
      excluded: own(previous, 'excluded') ? previous.excluded === true
        : ['experimenter', 'other'].includes(previous.speaker_role) || previous.text_decision === 'no_speech',
      notes: previous.notes ?? '',
    };
    if (previous.legacy_review) review.legacy_review = previous.legacy_review;
    else if (previous.text_decision || previous.speaker_role || previous.timing_decision)
      review.legacy_review = {...previous};
    if (previous.speaker_role === 'multiple' && !own(previous, 'text_checked')) review.text_checked = false;
    if (!String(review.human_text).trim()) review.text_checked = false;
    if (!validTimes(review, item, file)) review.timing_checked = false;
    return review;
  }
  function accepted(review, item, file) {
    return !review.excluded && review.text_checked && review.timing_checked
      && String(review.human_text).trim().length > 0 && validTimes(review, item, file);
  }
  function identity(item) {
    return {review_id: item.review_id, utterance_id: item.utterance_id,
      source_relative_path: item.source_relative_path, source_text_sha256: item.source_text_sha256};
  }
  function exportReview(data, state, uiLanguage) {
    const files = new Map(data.files.map(file => [file.source_relative_path, file]));
    const audit = data.items.map(item => ({...identity(item), ...state.items[item.review_id]}));
    const items = data.items.filter(item => accepted(state.items[item.review_id], item,
      files.get(item.source_relative_path))).map(item => ({...identity(item),
        human_text: state.items[item.review_id].human_text.trim(),
        human_start_audio_s: number(state.items[item.review_id].human_start_audio_s),
        human_end_audio_s: number(state.items[item.review_id].human_end_audio_s),
        text_checked: true, timing_checked: true}));
    return {schema: 'impact-audio-review', schema_version: 2, language: data.language,
      ui_language: uiLanguage, plan_hash: data.plan_hash, reviewer: state.reviewer || '',
      exported_at: new Date().toISOString(), items, review_records: audit,
      files: data.files.map(file => ({source_relative_path: file.source_relative_path,
        source_sha256: file.source_sha256, ...(state.files[file.source_relative_path] || {})}))};
  }
  function importReview(data, state, payload) {
    if (payload.plan_hash && payload.plan_hash !== data.plan_hash)
      throw new Error('plan_mismatch');
    const source = new Map(data.items.map(item => [item.review_id, item]));
    const files = new Map(data.files.map(file => [file.source_relative_path, file]));
    const records = payload.review_records || payload.items;
    if (!Array.isArray(records)) throw new Error('invalid_import');
    let imported = 0, skipped = 0;
    for (const record of records) {
      const item = source.get(record.review_id);
      if (!item || record.source_text_sha256 !== item.source_text_sha256
          || (record.source_relative_path && record.source_relative_path !== item.source_relative_path)) {
        skipped++; continue;
      }
      state.items[item.review_id] = normalize(item, record, files.get(item.source_relative_path));
      imported++;
    }
    for (const record of payload.files || []) {
      const file = files.get(record.source_relative_path);
      if (file && record.source_sha256 === file.source_sha256) {
        state.files[file.source_relative_path] = {review_state: record.review_state || 'pending',
          notes: record.notes || ''};
      }
    }
    if (payload.reviewer) state.reviewer = String(payload.reviewer);
    return {imported, skipped};
  }
  return {number, validTimes, normalize, accepted, exportReview, importReview};
})();
