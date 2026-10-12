(() => {
  'use strict';
  const $ = id => document.getElementById(id), C = AudioReviewCore;
  const files = new Map(DATA.files.map(file => [file.source_relative_path, file]));
  const storageKey = 'impact-review-' + DATA.language + '-' + DATA.plan_hash;
  let state = {items: {}, files: {}, reviewer: ''}, storageFailed = false;
  try { const prior = JSON.parse(localStorage.getItem(storageKey) || 'null'); if (prior) state = {...state, ...prior}; }
  catch (_) { storageFailed = true; }
  state.items = state.items || {}; state.files = state.files || {};
  DATA.items.forEach(item => state.items[item.review_id] = C.normalize(item,
    state.items[item.review_id] || DATA.initial_reviews?.[item.review_id], files.get(item.source_relative_path)));
  const itemIndex = new Map(DATA.items.map((item, index) => [item.review_id, index]));
  let current = Math.max(0, itemIndex.get(state.current_review_id) ?? 0), visible = [], clip = null;
  let timer = null, animation = null, generation = 0, activeMode = 'current';
  const player = $('player');
  document.querySelectorAll('[data-t]').forEach(node => node.textContent = L[node.dataset.t] || node.dataset.t);
  document.querySelectorAll('[data-ph]').forEach(node => node.placeholder = L[node.dataset.ph] || node.dataset.ph);
  $('reviewer').value = state.reviewer;
  function persist() {
    state.current_review_id = DATA.items[current]?.review_id;
    try { localStorage.setItem(storageKey, JSON.stringify(state)); storageFailed = false; }
    catch (_) { storageFailed = true; }
    $('saved').textContent = storageFailed ? L.saveError : L.saved;
    $('saved').classList.toggle('error', storageFailed);
  }
  function record() { return DATA.items[current]; }
  function review() { return state.items[record()?.review_id]; }
  function finishTimers() { clearTimeout(timer); timer = null; cancelAnimationFrame(animation); animation = null; }
  function stop() { generation++; finishTimers(); player.pause(); $('play').textContent = L.play; }
  function bounds(mode) {
    const item = record(), r = review(), file = files.get(item.source_relative_path);
    let start = mode === 'current' ? C.number(r.human_start_audio_s) : C.number(item.original_start_audio_s);
    let end = mode === 'current' ? C.number(r.human_end_audio_s) : C.number(item.original_end_audio_s);
    if (mode === 'context') { start = Math.max(0, start - 2); end = Math.min(file.duration_seconds, end + 2); }
    return {start, end, duration: file.duration_seconds, path: item.audio_file || file.audio_file};
  }
  function validClip(value) {
    return value && Number.isFinite(value.start) && Number.isFinite(value.end)
      && value.start >= 0 && value.end > value.start && value.end <= value.duration + 0.001;
  }
  const format = seconds => {
    seconds = Math.max(0, Number(seconds) || 0);
    return Math.floor(seconds / 60) + ':' + (seconds % 60).toFixed(2).padStart(5, '0');
  };
  function drawSeek() {
    if (!clip) return;
    const duration = clip.end - clip.start;
    const offset = Math.max(0, Math.min(duration, player.currentTime - clip.start));
    $('seek').max = duration; $('seek').value = offset;
    $('elapsed').textContent = format(offset); $('clip-duration').textContent = format(duration);
  }
  function enforceEnd() {
    if (!clip || player.paused) return;
    const remaining = clip.end - player.currentTime;
    if (remaining <= 0.012) {
      player.pause(); finishTimers();
      if (Math.abs(player.currentTime - clip.end) > 0.003) player.currentTime = clip.end;
      $('play').textContent = L.play; drawSeek(); return;
    }
    clearTimeout(timer);
    timer = setTimeout(enforceEnd, Math.max(5, Math.min(100, remaining / player.playbackRate * 1000)));
  }
  function animate() {
    drawSeek(); enforceEnd();
    if (!player.paused) animation = requestAnimationFrame(animate);
  }
  async function playClip(mode, offset = null) {
    stop(); const token = generation;
    delete player.dataset.playbackError;
    const value = bounds(mode);
    if (!validClip(value)) { $('validation').textContent = L.invalidTime; return; }
    clip = value; activeMode = mode;
    $('seek').max = value.end - value.start; $('seek').value = 0;
    $('elapsed').textContent = format(0); $('clip-duration').textContent = format(value.end - value.start);
    $('play-mode').textContent = mode === 'original' ? L.original : mode === 'context' ? L.context : L.current;
    const url = new URL(value.path, window.location.href).href;
    const base = player.src.split('#')[0];
    // Native media fragments bound playback even if the page is backgrounded.
    if (base !== url || player.dataset.fragment !== `${value.start},${value.end}`) {
      player.src = url + '#t=' + value.start + ',' + value.end;
      player.dataset.fragment = `${value.start},${value.end}`;
      player.load();
    }
    const position = offset == null ? value.start : Math.max(value.start, Math.min(value.end, offset));
    try {
      if (player.readyState < 1) await new Promise((resolve, reject) => {
        let timeout;
        const cleanup = () => { clearTimeout(timeout); player.removeEventListener('loadedmetadata', loaded); player.removeEventListener('error', failed); };
        const loaded = () => { cleanup(); resolve(); }, failed = () => { cleanup(); reject(new Error('audio')); };
        player.addEventListener('loadedmetadata', loaded); player.addEventListener('error', failed);
        timeout = setTimeout(failed, 12000);
      });
      if (token !== generation) return;
      if (value.end > player.duration + 0.001) throw new Error('duration');
      player.currentTime = position; player.playbackRate = Number($('rate').value);
      await player.play();
      if (token !== generation) return;
      $('audio-status').textContent = ''; $('play').textContent = L.pause;
      finishTimers(); animation = requestAnimationFrame(animate); enforceEnd();
    } catch (error) { if (token === generation) {
      player.dataset.playbackError = error.name || error.message;
      stop(); $('audio-status').textContent = L.playbackError;
    } }
  }
  function refreshList() {
    const path = $('recording').value, search = $('search').value.toLowerCase().trim();
    visible = DATA.items.map((item, index) => ({item, index})).filter(({item}) => {
      const r = state.items[item.review_id];
      return item.source_relative_path === path
        && (!search || (item.original_text + ' ' + r.human_text + ' ' + item.source_relative_path).toLowerCase().includes(search))
        && ($('filter').value !== 'pending' || (!r.excluded && !C.accepted(r, item, files.get(path))));
    }).map(({index}) => index);
    if (!visible.includes(current)) current = visible[0] ?? -1;
    render();
  }
  function progress() {
    let approved = 0, excluded = 0;
    DATA.items.forEach(item => { const r = state.items[item.review_id];
      if (r.excluded) excluded++; else if (C.accepted(r, item, files.get(item.source_relative_path))) approved++; });
    $('progress').textContent = `${L.accepted} ${approved} / ${DATA.items.length} · ${L.excluded} ${excluded}`;
    const r = review(), item = record();
    $('state').textContent = !r ? '' : r.excluded ? L.excluded
      : C.accepted(r, item, files.get(item.source_relative_path)) ? L.accepted : L.pendingState;
    $('text-ok').checked = r?.text_checked || false; $('time-ok').checked = r?.timing_checked || false;
    $('accept').disabled = !r || r.excluded;
  }
  function render() {
    stop(); clip = null; $('validation').textContent = ''; $('audio-status').textContent = '';
    const item = record(), r = review();
    for (const id of ['text', 'start', 'end', 'notes', 'text-ok', 'time-ok', 'excluded', 'play', 'original', 'context', 'evidence', 'set-start', 'set-end', 'seek']) $(id).disabled = !item;
    const displayTime = value => Number.isFinite(C.number(value)) ? Number(C.number(value).toFixed(3)) : value ?? '';
    $('text').value = r?.human_text ?? ''; $('start').value = displayTime(r?.human_start_audio_s);
    $('end').value = displayTime(r?.human_end_audio_s); $('notes').value = r?.notes ?? '';
    $('excluded').checked = r?.excluded || false;
    const position = visible.indexOf(current);
    $('position').textContent = item ? `${position + 1} / ${visible.length} · ${item.review_id}` : L.noItems;
    $('previous').disabled = position <= 0; $('next').disabled = position < 0 || position >= visible.length - 1;
    if (item) {
      const value = bounds('current');
      $('seek').max = validClip(value) ? value.end - value.start : 1; $('seek').value = 0;
      $('elapsed').textContent = format(0); $('clip-duration').textContent = validClip(value) ? format(value.end - value.start) : '—';
      $('play-mode').textContent = L.current;
    }
    progress(); persist();
  }
  function navigate(direction) {
    const position = visible.indexOf(current), next = position + direction;
    if (next >= 0 && next < visible.length) { current = visible[next]; render(); }
  }
  function approve() {
    const item = record(), r = review(); if (!r || r.excluded) return;
    if (!String(r.human_text).trim()) { $('validation').textContent = L.emptyText; return; }
    if (!C.validTimes(r, item, files.get(item.source_relative_path))) { $('validation').textContent = L.invalidTime; return; }
    r.text_checked = true; r.timing_checked = true; persist(); progress();
    if ($('filter').value === 'pending') {
      const next = visible[visible.indexOf(current) + 1] ?? visible[0];
      current = next; refreshList();
    } else if (visible.indexOf(current) < visible.length - 1) navigate(1);
    else $('validation').textContent = L.finished;
  }
  function fieldChanged(field, value) {
    const r = review(); if (!r) return;
    r[field] = value;
    if (field === 'human_text') r.text_checked = false;
    if (field === 'human_start_audio_s' || field === 'human_end_audio_s') {
      r.timing_checked = false; stop(); clip = null;
      const value = bounds('current'); $('seek').value = 0; $('elapsed').textContent = format(0);
      $('clip-duration').textContent = validClip(value) ? format(value.end - value.start) : '—';
    }
    $('validation').textContent = ''; persist(); progress();
  }
  $('text').oninput = () => fieldChanged('human_text', $('text').value);
  $('start').oninput = () => fieldChanged('human_start_audio_s', $('start').value);
  $('end').oninput = () => fieldChanged('human_end_audio_s', $('end').value);
  $('notes').oninput = () => fieldChanged('notes', $('notes').value);
  $('text-ok').onchange = () => {
    const r = review(); r.text_checked = $('text-ok').checked && !!String(r.human_text).trim();
    $('validation').textContent = $('text-ok').checked && !r.text_checked ? L.emptyText : ''; persist(); progress();
  };
  $('time-ok').onchange = () => {
    const r = review(); r.timing_checked = $('time-ok').checked && C.validTimes(r, record(), files.get(record().source_relative_path));
    $('validation').textContent = $('time-ok').checked && !r.timing_checked ? L.invalidTime : ''; persist(); progress();
  };
  $('excluded').onchange = () => { review().excluded = $('excluded').checked; persist(); progress(); };
  $('reviewer').oninput = () => { state.reviewer = $('reviewer').value; persist(); };
  $('previous').onclick = () => navigate(-1); $('next').onclick = () => navigate(1); $('accept').onclick = approve;
  $('play').onclick = () => {
    if (!player.paused) { stop(); }
    else {
      const resume = clip && activeMode === 'current' && player.currentTime >= clip.start && player.currentTime < clip.end - .02 ? player.currentTime : null;
      playClip('current', resume);
    }
  };
  $('original').onclick = () => playClip('original'); $('context').onclick = () => playClip('context');
  $('seek').oninput = () => {
    if (!clip) { playClip('current', C.number(review().human_start_audio_s) + Number($('seek').value)); return; }
    player.currentTime = Math.max(clip.start, Math.min(clip.end, clip.start + Number($('seek').value)));
    drawSeek(); enforceEnd();
  };
  for (const [id, field] of [['set-start', 'human_start_audio_s'], ['set-end', 'human_end_audio_s']]) $(id).onclick = () => {
    if (!clip || !Number.isFinite(player.currentTime)) return;
    const value = Number(player.currentTime.toFixed(3));
    fieldChanged(field, value); $(field.includes('start') ? 'start' : 'end').value = value;
  };
  $('rate').onchange = () => { player.playbackRate = Number($('rate').value); enforceEnd(); };
  player.addEventListener('timeupdate', () => { drawSeek(); enforceEnd(); });
  player.addEventListener('playing', enforceEnd);
  player.addEventListener('pause', () => { finishTimers(); $('play').textContent = L.play; drawSeek(); });
  player.addEventListener('ended', () => { finishTimers(); $('play').textContent = L.play; drawSeek(); });
  player.addEventListener('seeking', () => {
    if (!clip) return;
    if (player.currentTime < clip.start) player.currentTime = clip.start;
    else if (player.currentTime > clip.end) player.currentTime = clip.end;
  });
  player.addEventListener('error', () => { if (clip) $('audio-status').textContent = L.playbackError; });
  DATA.files.forEach(file => { const option = document.createElement('option'); option.value = file.source_relative_path;
    option.textContent = file.source_relative_path.split('/').pop(); $('recording').append(option); });
  $('recording').value = record()?.source_relative_path || DATA.files[0]?.source_relative_path || '';
  $('recording').onchange = refreshList; $('search').oninput = refreshList; $('filter').onchange = refreshList;
  function details(title, text) { $('details-title').textContent = title; $('details-body').textContent = text; $('details').showModal(); }
  $('help').onclick = () => details(L.help, L.instructions);
  $('close-details').onclick = () => $('details').close();
  $('evidence').onclick = () => {
    const item = record();
    const parts = [`${L.originalText}\n${item.original_text}`, `ASR: ${item.original_start_audio_s} – ${item.original_end_audio_s} s`];
    for (const key of ['offline_review_context', 'machine_text_clip_with_context', 'alternate_machine_text', 'literal_time_proposal', 'machine_word_times_display'])
      if (item[key]) parts.push(`${key}\n${typeof item[key] === 'string' ? item[key] : JSON.stringify(item[key], null, 2)}`);
    const fileNotes = state.files[item.source_relative_path]?.notes;
    if (fileNotes) parts.push(`${L.notes}\n${fileNotes}`);
    details(L.evidence, parts.join('\n\n'));
  };
  $('export').onclick = () => {
    persist();
    try {
      const payload = C.exportReview(DATA, state, L.uiLanguage);
      const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'impact_audio_review_' + DATA.language + '.json';
      document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
      $('audio-status').textContent = L.exportStatus.replace('{n}', payload.items.length);
    } catch (_) { $('audio-status').textContent = L.saveBlocked; }
  };
  $('import').onclick = () => $('import-file').click();
  $('import-file').onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      const result = C.importReview(DATA, state, JSON.parse(await file.text()));
      $('reviewer').value = state.reviewer; persist(); refreshList();
      $('audio-status').textContent = L.imported.replace('{n}', result.imported).replace('{s}', result.skipped);
    } catch (_) { $('audio-status').textContent = L.importError; }
    event.target.value = '';
  };
  document.addEventListener('keydown', event => {
    if ($('details').open) return;
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); approve(); return; }
    if (event.altKey && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) { event.preventDefault(); navigate(event.key === 'ArrowLeft' ? -1 : 1); return; }
    if (event.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(event.target.tagName)) { event.preventDefault(); $('play').click(); }
  });
  window.addEventListener('beforeunload', persist);
  // Small read API for integrations; edits continue through the review controls.
  window.IMPACTAudioReview = {core: C, data: DATA,
    exportReview: () => C.exportReview(DATA, state, L.uiLanguage)};
  refreshList();
})();
