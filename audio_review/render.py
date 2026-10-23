"""Render a standalone offline audio review page. Never modify ASR source records."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent
LABELS = {
    'zh': dict(reviewer='审核人', import_='导入记录', export='保存审核文件', search='搜索文字／文件名',
        all='全部句子', pending='未完成', help='说明', play='播放当前句', pause='暂停',
        original='原 ASR 片段', context='前后 2 秒', speed='速度', start='起点', end='终点',
        setStart='设为播放位置', setEnd='设为播放位置', seconds='原音频秒数', text='转写文字',
        textOk='文字正确', timeOk='时间正确', exclude='排除该句（非执行人／无有效语音）',
        notes='备注（可选）', previous='上一句', next='下一句', accept='通过并下一句',
        keys='空格播放／暂停 · Alt+←/→ 切换 · Ctrl/⌘+Enter 通过', evidence='原文与参考', close='关闭',
        saved='已自动保存', saveError='自动保存失败，请保存审核文件', accepted='已通过', excluded='已排除',
        pendingState='待审核', invalidTime='时间须满足：0 ≤ 起点 < 终点 ≤ 音频长度。',
        emptyText='请填写有效文字，或排除该句。', noItems='没有匹配的句子', current='当前片段',
        playbackError='音频无法播放，请确认已下载配套 audio 文件夹。', importError='无法导入：文件格式或任务来源不符。',
        imported='已导入 {n} 句，跳过 {s} 条不匹配记录。', exportStatus='已生成审核文件；正式转写包含 {n} 句。',
        saveBlocked='审核文件未保存，请重试。', finished='已到最后一句。', originalText='原始 ASR 文字',
        instructions='逐句播放并核对文字和起止时间。正确就勾选；有误直接修改，再勾选。\n“通过并下一句”同时确认文字和时间。非执行人的话勾选“排除该句”。\n起止时间已按 ASR 自动填入，单位是原音频秒数。播放只覆盖当前句；需要调整边界时可听“前后 2 秒”。\n修改文字会取消文字确认；修改起止时间会取消时间确认。\n编辑自动保存在本机浏览器。每次结束请保存审核文件：正式转写只含已通过的句子，文件同时保留全部审核进度，可再次导入。\n空格：播放／暂停；Alt+←/→：上一句／下一句；Ctrl/⌘+Enter：通过并下一句。'),
    'en': dict(reviewer='Reviewer', import_='Import review', export='Save review file', search='Search text / filename',
        all='All utterances', pending='Unfinished', help='Help', play='Play utterance', pause='Pause',
        original='Original ASR clip', context='±2 seconds', speed='Speed', start='Start', end='End',
        setStart='Use playhead', setEnd='Use playhead', seconds='Original audio seconds', text='Transcript',
        textOk='Text correct', timeOk='Timing correct', exclude='Exclude (not the operator / no speech)',
        notes='Notes (optional)', previous='Previous', next='Next', accept='Approve & next',
        keys='Space: play/pause · Alt+←/→: navigate · Ctrl/⌘+Enter: approve', evidence='Original & references', close='Close',
        saved='Autosaved', saveError='Autosave failed. Save a review file.', accepted='Approved', excluded='Excluded',
        pendingState='To review', invalidTime='Use: 0 ≤ start < end ≤ recording duration.',
        emptyText='Enter speech text, or exclude this utterance.', noItems='No matching utterances', current='Edited clip',
        playbackError='Cannot play audio. Download the accompanying audio folder.', importError='Cannot import: invalid format or different task source.',
        imported='Imported {n} utterances; skipped {s} unmatched records.', exportStatus='Review file created; final transcript contains {n} utterances.',
        saveBlocked='Review file could not be saved. Please retry.', finished='Last utterance reached.', originalText='Original ASR text',
        instructions='Listen to each utterance and check its text and boundaries. Tick what is correct; edit errors, then tick.\n“Approve & next” confirms both text and timing. Exclude speech from anyone other than the operator.\nStart and end are filled from ASR in original audio seconds. Playback stops at the utterance end; use “±2 seconds” to inspect boundaries.\nEditing text clears its approval. Editing boundaries clears timing approval.\nEdits autosave in this browser. Save a review file at the end of each session: it includes approved transcript entries and all review progress for re-import.\nSpace: play/pause; Alt+←/→: previous/next; Ctrl/⌘+Enter: approve & next.'),
    'de': dict(reviewer='Prüfer/in', import_='Prüfung importieren', export='Prüfung speichern', search='Text / Dateiname suchen',
        all='Alle Äußerungen', pending='Offen', help='Hilfe', play='Äußerung abspielen', pause='Pause',
        original='Original-ASR-Ausschnitt', context='±2 Sekunden', speed='Tempo', start='Anfang', end='Ende',
        setStart='Abspielposition übernehmen', setEnd='Abspielposition übernehmen', seconds='Sekunden der Originalaufnahme', text='Transkript',
        textOk='Text korrekt', timeOk='Zeitgrenzen korrekt', exclude='Ausschließen (andere Person / keine Sprache)',
        notes='Notiz (optional)', previous='Zurück', next='Weiter', accept='Bestätigen & weiter',
        keys='Leertaste: Wiedergabe · Alt+←/→: wechseln · Strg/⌘+Enter: bestätigen', evidence='Original & Hinweise', close='Schließen',
        saved='Automatisch gespeichert', saveError='Automatisches Speichern fehlgeschlagen. Prüfdatei speichern.', accepted='Bestätigt', excluded='Ausgeschlossen',
        pendingState='Offen', invalidTime='Erforderlich: 0 ≤ Anfang < Ende ≤ Aufnahmelänge.',
        emptyText='Sprachtext eintragen oder Äußerung ausschließen.', noItems='Keine passenden Äußerungen', current='Bearbeiteter Ausschnitt',
        playbackError='Audio nicht verfügbar. Den zugehörigen audio-Ordner herunterladen.', importError='Import fehlgeschlagen: ungültiges Format oder andere Datenquelle.',
        imported='{n} Äußerungen importiert; {s} unpassende Einträge übersprungen.', exportStatus='Prüfdatei erstellt; das fertige Transkript enthält {n} Äußerungen.',
        saveBlocked='Prüfdatei konnte nicht gespeichert werden. Bitte erneut versuchen.', finished='Letzte Äußerung erreicht.', originalText='Original-ASR-Text',
        instructions='Jede Äußerung anhören und Text sowie Zeitgrenzen prüfen. Korrektes abhaken; Fehler bearbeiten und danach abhaken.\n„Bestätigen & weiter“ bestätigt Text und Zeitgrenzen. Sprache anderer Personen als der ausführenden Person ausschließen.\nAnfang und Ende sind aus ASR ausgefüllt, in Sekunden der Originalaufnahme. Die Wiedergabe endet am Satzende. Mit „±2 Sekunden“ die Grenzen prüfen.\nTextänderungen setzen die Textbestätigung zurück; Zeitänderungen setzen die Zeitbestätigung zurück.\nÄnderungen werden in diesem Browser gespeichert. Am Sitzungsende eine Prüfdatei speichern: Sie enthält das bestätigte Transkript und den gesamten Prüfstand für den erneuten Import.\nLeertaste: Wiedergabe/Pause; Alt+←/→: zurück/weiter; Strg/⌘+Enter: bestätigen & weiter.')
}
for _lang, _values in {
    'zh': dict(gain='音量增益', audioFolder='启用增益：选择 audio 文件夹', boostSelect='请先选择配套 audio 文件夹，再使用增益。', boostReady='增益已就绪，仅影响此工具。', originalClock='原音频位置', exclude='删除／排除该句（可撤销）', instructionsExtra='\n删除片段：勾选“删除／排除该句”。该句不进入正式转写；取消勾选即可恢复。\n音量太低：点击“启用增益”，选择配套 audio 文件夹，然后调高增益。只提高本工具的音量，无需调高电脑总音量。'),
    'en': dict(gain='Audio boost', audioFolder='Enable boost: select audio folder', boostSelect='Select the accompanying audio folder to enable boost.', boostReady='Boost ready. Only this tool is affected.', originalClock='Recording position', exclude='Delete / exclude utterance (undoable)', instructionsExtra='\nDelete a segment by ticking “Delete / exclude utterance”. It is omitted from the final transcript; untick to restore it.\nFor quiet audio, enable boost by selecting the accompanying audio folder, then increase Audio boost. This does not change the computer volume.'),
    'de': dict(gain='Verstärkung', audioFolder='Verstärkung: audio-Ordner wählen', boostSelect='Zum Verstärken zuerst den zugehörigen audio-Ordner wählen.', boostReady='Verstärkung bereit. Gilt nur für dieses Werkzeug.', originalClock='Aufnahmeposition', exclude='Äußerung löschen / ausschließen (rückgängig)', instructionsExtra='\nEine Äußerung mit „löschen / ausschließen“ aus dem fertigen Transkript entfernen. Zum Wiederherstellen das Häkchen entfernen.\nBei leiser Aufnahme den audio-Ordner wählen und die Verstärkung erhöhen. Die Systemlautstärke bleibt unverändert.')
}.items():
    LABELS[_lang].update(_values)
    LABELS[_lang]['instructions'] += LABELS[_lang].pop('instructionsExtra')

for _labels in LABELS.values():
    _labels['import'] = _labels.pop('import_')

def _script_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')

def render_page(data, ui_language=None):
    ui_language = ui_language or data.get('ui_language') or data['language']
    if ui_language not in LABELS:
        raise ValueError('Unsupported interface language')
    labels = dict(LABELS[ui_language], uiLanguage=ui_language)
    replacements = {
        '__UI_LANGUAGE__': ui_language, '__STYLE__': (ROOT / 'review.css').read_text(encoding='utf-8'),
        '__DATA__': _script_json(data), '__LABELS__': _script_json(labels),
        '__CORE__': (ROOT / 'review_core.js').read_text(encoding='utf-8'), '__APP__': (ROOT / 'review_app.js').read_text(encoding='utf-8'),
    }
    # A single replacement pass prevents transcript text resembling a placeholder
    # from being treated as part of the template.
    import re
    return re.sub('|'.join(map(re.escape, replacements)), lambda m: replacements[m[0]],
                  (ROOT / 'review_template.html').read_text(encoding='utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ui-language', choices=sorted(LABELS))
    args = parser.parse_args()
    args.output.write_text(render_page(json.loads(args.data.read_text(encoding='utf-8')), args.ui_language), encoding='utf-8')
