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
  C.restoreAdded(DATA,state);
  let items=C.allItems(DATA,state);
  const itemIndex = new Map(items.map((item, index) => [item.review_id, index]));
  let current = Math.max(0, itemIndex.get(state.current_review_id) ?? 0), visible = [], clip = null;
  let timer = null, animation = null, generation = 0, activeMode = 'current';
  const player = $('player');
  let audioContext = null, boostNode = null, compressor = null;
  const localAudio = new Map(), blobURLs = new Map();
  function localName(path) { return decodeURIComponent(path.split('/').pop()); }
  function playbackURL(path) {
    const name = localName(path), file = localAudio.get(name);
    if (audioContext || Number($('gain').value) > 1) {
      if (!file) throw new Error('boost-folder');
      if (!blobURLs.has(name)) blobURLs.set(name, URL.createObjectURL(file));
      return blobURLs.get(name);
    }
    return new URL(path, window.location.href).href;
  }
  async function prepareBoost() {
    if (Number($('gain').value) <= 1 && !audioContext) return;
    if (!audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) throw new Error('boost-folder');
      audioContext = new AudioCtx();
      const source = audioContext.createMediaElementSource(player);
      boostNode = audioContext.createGain(); compressor = audioContext.createDynamicsCompressor();
      compressor.threshold.value = -6; compressor.knee.value = 12; compressor.ratio.value = 12;
      compressor.attack.value = 0.003; compressor.release.value = 0.15;
      source.connect(boostNode); boostNode.connect(compressor); compressor.connect(audioContext.destination);
    }
    boostNode.gain.value = Number($('gain').value);
    await audioContext.resume();
  }
  $('audio-folder').onclick = () => $('audio-folder-files').click();
  $('audio-folder-files').onchange = event => {
    stop();
    const duplicates = new Set(); localAudio.clear();
    for (const file of event.target.files) {
      if (!/\.(wav|mp3|m4a|ogg|flac|aac|opus|mp4)$/i.test(file.name)) continue;
      if (localAudio.has(file.name)) duplicates.add(file.name); else localAudio.set(file.name, file);
    }
    for (const name of duplicates) localAudio.delete(name);
    for (const url of blobURLs.values()) URL.revokeObjectURL(url); blobURLs.clear();
    $('audio-status').textContent = L.boostReady;
  };
  $('gain').oninput = () => {
    $('gain-value').textContent = $('gain').value + '×';
    if (boostNode) boostNode.gain.setTargetAtTime(Number($('gain').value), audioContext.currentTime, .03);
    else if (Number($('gain').value) > 1) { stop(); $('audio-status').textContent = L.boostSelect; }
  };
  document.querySelectorAll('[data-t]').forEach(node => node.textContent = L[node.dataset.t] || node.dataset.t);
  document.querySelectorAll('[data-ph]').forEach(node => node.placeholder = L[node.dataset.ph] || node.dataset.ph);
  $('reviewer').value = state.reviewer;
  function persist() {
    state.current_review_id = items[current]?.review_id;
    try { localStorage.setItem(storageKey, JSON.stringify(state)); storageFailed = false; }
    catch (_) { storageFailed = true; }
    $('saved').textContent = storageFailed ? L.saveError : L.saved;
    $('saved').classList.toggle('error', storageFailed);
  }
  function record() { return items[current]; }
  function review() { return state.items[record()?.review_id]; }
  function finishTimers() { clearTimeout(timer); timer = null; cancelAnimationFrame(animation); animation = null; }
  function stop() { generation++; finishTimers(); player.pause(); $('play').textContent = L.play; }
  function bounds(mode) {
    const item = record(), r = review(), file = files.get(item?.source_relative_path || $('recording').value);
    if (mode === 'recording') return {start:0,end:file?.duration_seconds,duration:file?.duration_seconds,path:file?.audio_file};
    let start = (mode === 'current' || item.annotation_origin==='human_added') ? C.number(r.human_start_audio_s) : C.number(item.original_start_audio_s);
    let end = (mode === 'current' || item.annotation_origin==='human_added') ? C.number(r.human_end_audio_s) : C.number(item.original_end_audio_s);
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
    $('absolute-time').textContent = format(player.currentTime);
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
    $('play-mode').textContent = mode === 'recording' ? L.recordingPlay : mode === 'original' ? L.original : mode === 'context' ? L.context : L.current;
    let url;
    try { url = playbackURL(value.path); }
    catch (_) { $('audio-status').textContent = L.boostSelect; return; }
    const base = player.src.split('#')[0];
    // Native media fragments bound playback even if the page is backgrounded.
    if (base !== url || player.dataset.fragment !== `${value.start},${value.end}`) {
      player.src = url + '#t=' + value.start + ',' + value.end;
      player.dataset.fragment = `${value.start},${value.end}`;
      player.load();
    }
    const position = offset == null ? value.start : Math.max(value.start, Math.min(value.end, offset));
    try {
      // Resume Web Audio during the play-button gesture, before metadata awaits.
      const readyForBoost = prepareBoost();
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
      await readyForBoost;
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
    visible = items.map((item, index) => ({item, index})).filter(({item}) => {
      const r = state.items[item.review_id];
      return item.source_relative_path === path
        && (!search || (item.original_text + ' ' + r.human_text + ' ' + item.source_relative_path).toLowerCase().includes(search))
        && ($('filter').value !== 'pending' || (!r.excluded && !C.accepted(r, item, files.get(path))));
    }).map(({index}) => index);
    if (!visible.includes(current)) current = visible[0] ?? -1;
    sortVisible(); render();
  }
  function sortVisible() {
    visible.sort((a,b)=>(C.number(state.items[items[a].review_id].human_start_audio_s)||0)-(C.number(state.items[items[b].review_id].human_start_audio_s)||0) || a-b);
  }
  function progress() {
    let approved = 0, excluded = 0;
    items.forEach(item => { const r = state.items[item.review_id];
      if (r.excluded) excluded++; else if (C.accepted(r, item, files.get(item.source_relative_path))) approved++; });
    $('progress').textContent = `${L.accepted} ${approved} / ${items.length} · ${L.excluded} ${excluded}`;
    const r = review(), item = record();
    $('state').textContent = !r ? '' : r.excluded ? L.excluded
      : C.accepted(r, item, files.get(item.source_relative_path)) ? L.accepted : L.pendingState;
    $('text-ok').checked = r?.text_checked || false; $('time-ok').checked = r?.timing_checked || false;
    $('accept').disabled = !r || r.excluded;
    sortVisible();const position=visible.indexOf(current);
    $('position').textContent = item ? `${position+1} / ${visible.length} · ${item.annotation_origin==='human_added' ? L.humanAdded : item.review_id}` : L.noItems;
    $('previous').disabled=position<=0; $('next').disabled=position<0||position>=visible.length-1;
  }
  function render() {
    stop(); clip = null; $('validation').textContent = ''; $('audio-status').textContent = '';
    const item = record(), r = review();
    for (const id of ['text', 'start', 'end', 'notes', 'text-ok', 'time-ok', 'excluded', 'play', 'original', 'context', 'evidence', 'set-start', 'set-end', 'seek']) $(id).disabled = !item;
    const displayTime = value => Number.isFinite(C.number(value)) ? Number(C.number(value).toFixed(3)) : value ?? '';
    $('text').value = r?.human_text ?? ''; $('start').value = displayTime(r?.human_start_audio_s);
    $('end').value = displayTime(r?.human_end_audio_s); $('notes').value = r?.notes ?? '';
    $('excluded').checked = r?.excluded || false;
    $('original').disabled=!item || item.annotation_origin==='human_added';
    $('add').disabled=!files.has($('recording').value);
    $('recording-play').disabled=!files.has($('recording').value);
    const position = visible.indexOf(current);
    $('position').textContent = item ? `${position + 1} / ${visible.length} · ${item.review_id}` : L.noItems;
    $('previous').disabled = position <= 0; $('next').disabled = position < 0 || position >= visible.length - 1;
    if (item) {
      const value = bounds('current');
      $('seek').max = validClip(value) ? value.end - value.start : 1; $('seek').value = 0;
      $('elapsed').textContent = format(0); $('clip-duration').textContent = validClip(value) ? format(value.end - value.start) : '—';
      $('play-mode').textContent = L.current;
      $('absolute-time').textContent = validClip(value) ? format(value.start) : '—';
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
      r.timing_checked = false;
      if (activeMode==='recording' && clip) { $('validation').textContent='';persist();progress();return; }
      stop(); clip = null;
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
  $('recording-play').onclick = () => {
    if (activeMode==='recording' && !player.paused) stop();
    else playClip('recording', Number.isFinite(C.number(review()?.human_start_audio_s)) ? C.number(review().human_start_audio_s) : 0);
  };
  $('add').onclick = () => {
    const path=$('recording').value,file=files.get(path),duration=C.number(file?.duration_seconds);
    if (!(duration>0)) { $('validation').textContent=L.invalidTime;return; }
    let start=clip && record()?.source_relative_path===path ? player.currentTime : C.number(review()?.human_end_audio_s);
    if (!Number.isFinite(start)) start=0;
    start=Math.max(0,Math.min(start,Math.max(0,duration-0.01)));
    let end=Math.min(duration,start+2);
    const next=items.map(i=>state.items[i.review_id]).filter((r,index)=>items[index].source_relative_path===path && !r.excluded).map(r=>C.number(r.human_start_audio_s)).filter(t=>t>start+.01).sort((a,b)=>a-b)[0];
    if (next!=null) end=Math.min(end,next);
    const bytes=crypto.getRandomValues(new Uint8Array(16));
    const id='human-'+Array.from(bytes,x=>x.toString(16).padStart(2,'0')).join('');
    try { C.addItem(DATA,state,path,start,end,id);items=C.allItems(DATA,state);current=items.findIndex(i=>i.review_id===id);$('search').value='';refreshList();$('text').focus(); }
    catch (_) { $('validation').textContent=L.invalidTime; }
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
    const parts = item.annotation_origin==='human_added' ? [L.humanAdded] : [`${L.originalText}\n${item.original_text}`, `ASR: ${item.original_start_audio_s} – ${item.original_end_audio_s} s`];
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
      items=C.allItems(DATA,state);$('reviewer').value = state.reviewer; persist(); refreshList();
      $('audio-status').textContent = L.imported.replace('{n}', result.imported).replace('{s}', result.skipped);
    } catch (_) { $('audio-status').textContent = L.importError; }
    event.target.value = '';
  };
  document.addEventListener('keydown', event => {
    if ($('details').open) return;
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); approve(); return; }
    if (event.altKey && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) { event.preventDefault(); navigate(event.key === 'ArrowLeft' ? -1 : 1); return; }
    if (event.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) { event.preventDefault(); $('play').click(); }
  });
  window.addEventListener('beforeunload', persist);
  // Small read API for integrations; edits continue through the review controls.
  window.IMPACTAudioReview = {core: C, data: DATA,
    exportReview: () => C.exportReview(DATA, state, L.uiLanguage)};
  refreshList();
})();
