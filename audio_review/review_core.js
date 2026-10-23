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
      source_relative_path: item.source_relative_path, source_text_sha256: item.source_text_sha256,
      ...(item.annotation_origin === 'human_added' ? {annotation_origin:'human_added',source_sha256:item.source_sha256} : {})};
  }
  function manualItem(data, candidate) {
    if (!candidate || !/^human-[A-Za-z0-9_-]{1,100}$/.test(candidate.review_id || '')) return null;
    const file = data.files.find(f => f.source_relative_path === candidate.source_relative_path);
    if (!file || !file.source_sha256 || candidate.source_sha256 !== file.source_sha256) return null;
    const start = number(candidate.original_start_audio_s), end = number(candidate.original_end_audio_s);
    if (!Number.isFinite(number(file.duration_seconds)) || !validTimes({human_start_audio_s:start,human_end_audio_s:end}, {}, file)) return null;
    return {review_id:candidate.review_id, utterance_id:candidate.review_id,
      source_relative_path:file.source_relative_path, source_sha256:file.source_sha256,
      source_text_sha256:null, annotation_origin:'human_added', original_text:'',
      original_start_audio_s:start, original_end_audio_s:end, audio_file:file.audio_file,
      created_at: typeof candidate.created_at === 'string' ? candidate.created_at : null};
  }
  function restoreAdded(data, state) {
    const reserved = new Set(data.items.map(i=>i.review_id)), result=[];
    for (const candidate of Array.isArray(state.added_items) ? state.added_items : []) {
      const item=manualItem(data,candidate);
      if (!item || reserved.has(item.review_id)) continue;
      reserved.add(item.review_id);result.push(item);
      state.items[item.review_id]=normalize(item,state.items[item.review_id],data.files.find(f=>f.source_relative_path===item.source_relative_path));
    }
    state.added_items=result;
    return result;
  }
  function allItems(data,state) { return data.items.concat(state.added_items || []); }
  function addItem(data,state,path,start,end,id) {
    const file=data.files.find(f=>f.source_relative_path===path);
    const item=manualItem(data,{review_id:id,source_relative_path:path,source_sha256:file?.source_sha256,
      original_start_audio_s:start,original_end_audio_s:end,created_at:new Date().toISOString()});
    if (!item || allItems(data,state).some(i=>i.review_id===id)) throw new Error('invalid_added_utterance');
    state.added_items=state.added_items || [];state.added_items.push(item);
    state.items[id]=normalize(item,{},file);
    return item;
  }
  function exportReview(data, state, uiLanguage) {
    const files = new Map(data.files.map(file => [file.source_relative_path, file]));
    const combined=allItems(data,state);
    const audit = combined.map(item => ({...identity(item), ...state.items[item.review_id]}));
    const order=new Map(data.files.map((f,i)=>[f.source_relative_path,i]));
    const items = combined.filter(item => accepted(state.items[item.review_id], item,
      files.get(item.source_relative_path))).map(item => ({...identity(item),
        human_text: state.items[item.review_id].human_text.trim(),
        human_start_audio_s: number(state.items[item.review_id].human_start_audio_s),
        human_end_audio_s: number(state.items[item.review_id].human_end_audio_s),
        text_checked: true, timing_checked: true})).sort((a,b)=>order.get(a.source_relative_path)-order.get(b.source_relative_path) || a.human_start_audio_s-b.human_start_audio_s);
    return {schema: 'impact-audio-review', schema_version: 3, language: data.language,
      ui_language: uiLanguage, plan_hash: data.plan_hash, reviewer: state.reviewer || '',
      exported_at: new Date().toISOString(), items, review_records: audit,
      added_items: (state.added_items || []).map(item=>({...item})),
      files: data.files.map(file => ({source_relative_path: file.source_relative_path,
        source_sha256: file.source_sha256, ...(state.files[file.source_relative_path] || {})}))};
  }
  function importReview(data, state, payload) {
    if (payload.plan_hash && payload.plan_hash !== data.plan_hash)
      throw new Error('plan_mismatch');
    const source = new Map(allItems(data,state).map(item => [item.review_id, item]));
    const files = new Map(data.files.map(file => [file.source_relative_path, file]));
    const records = payload.review_records || payload.items;
    if (!Array.isArray(records)) throw new Error('invalid_import');
    let imported = 0, skipped = 0;
    const rejected = new Set();
    for (const candidate of Array.isArray(payload.added_items) ? payload.added_items : []) {
      const item=payload.plan_hash===data.plan_hash ? manualItem(data,candidate) : null;
      const existing=source.get(candidate?.review_id);
      if (!item || (existing && (existing.annotation_origin!=='human_added' || existing.source_relative_path!==item.source_relative_path || existing.source_sha256!==item.source_sha256 || existing.original_start_audio_s!==item.original_start_audio_s || existing.original_end_audio_s!==item.original_end_audio_s))) {
        if (candidate?.review_id) rejected.add(candidate.review_id);
        skipped++;continue;
      }
      if (!existing) {
        state.added_items=state.added_items || [];state.added_items.push(item);
        state.items[item.review_id]=normalize(item,{},files.get(item.source_relative_path));source.set(item.review_id,item);
      }
    }
    for (const record of records) {
      const item = source.get(record.review_id);
      if (rejected.has(record.review_id) || !item || (item.annotation_origin==='human_added' && record.source_sha256!==item.source_sha256) || record.source_text_sha256 !== item.source_text_sha256
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
  return {number, validTimes, normalize, accepted, manualItem, restoreAdded, allItems, addItem, exportReview, importReview};
})();
