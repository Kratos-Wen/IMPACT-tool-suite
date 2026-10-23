"""Offline, real-browser tests. Uses synthetic audio only; no private transcripts."""
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
import wave

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audio_render', ROOT / 'audio_review/render.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class AudioReviewBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        for name in ('one.wav', 'two.wav'):
            with wave.open(str(cls.folder / name), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(b''.join(struct.pack('<h', int(800 * math.sin(i * .35))) for i in range(48000)))
        cls.data = {
            'language': 'en', 'ui_language': 'zh', 'plan_hash': 'synthetic-test-task',
            'files': [dict(source_relative_path=name, source_sha256='audio-' + name,
                           duration_seconds=6, audio_file=name) for name in ('one.wav', 'two.wav')],
            'items': [dict(review_id='u' + str(i), utterance_id='test:' + str(i), source_relative_path=name,
                           source_text_sha256='text-' + str(i), original_text=text,
                           original_start_audio_s=start, original_end_audio_s=end, audio_file=name)
                      for i, (name, text, start, end) in enumerate([
                          ('one.wav', 'Synthetic utterance one.', .35, .8),
                          ('one.wav', 'Synthetic utterance two.', 1.2, 1.8),
                          ('two.wav', 'Synthetic second recording.', 2.2, 2.7)])],
        }
        for language in ('zh', 'en', 'de'):
            (cls.folder / (language + '.html')).write_text(renderer.render_page(cls.data, language), encoding='utf-8')
        if os.environ.get('IMPACT_AUDIO_BROWSER') == 'chrome':
            options = webdriver.ChromeOptions()
            options.add_argument('--headless=new')
            options.add_argument('--disable-gpu')
            options.add_argument('--autoplay-policy=no-user-gesture-required')
            cls.browser = webdriver.Chrome(options=options)
        else:
            options = webdriver.FirefoxOptions()
            options.add_argument('-headless')
            if os.environ.get('IMPACT_FIREFOX_BINARY'):
                options.binary_location = os.environ['IMPACT_FIREFOX_BINARY']
            options.set_preference('media.autoplay.default', 0)
            options.set_preference('media.autoplay.blocking_policy', 0)
            options.set_preference('media.autoplay.block-webaudio', False)
            service = FirefoxService(executable_path=os.environ['IMPACT_GECKODRIVER']) if os.environ.get('IMPACT_GECKODRIVER') else FirefoxService()
            cls.browser = webdriver.Firefox(options=options, service=service)
        cls.browser.set_window_size(1366, 768)
        cls.wait = WebDriverWait(cls.browser, 12)

    @classmethod
    def tearDownClass(cls):
        cls.browser.quit()
        cls.temp.cleanup()

    def setUp(self):
        data = json.loads(json.dumps(self.data))
        data['plan_hash'] = self.id()
        # Each test gets its own file URL. Replacing one file repeatedly can
        # reuse cached HTML/form state across unrelated Chrome navigations.
        page = self.folder / (self._testMethodName + '.html')
        page.write_text(renderer.render_page(data, 'zh'), encoding='utf-8')
        self.browser.get(page.as_uri())
        self.wait_for_page()
        self.js('document.getElementById("player").muted=true;')

    def wait_for_page(self):
        self.wait.until(lambda driver: driver.execute_script('''
          return document.readyState === 'complete'
            && window.IMPACTAudioReview?.data.plan_hash === arguments[0];''', self.id()))

    def click(self, name):
        self.browser.find_element(By.ID, name).click()

    def js(self, script):
        return self.browser.execute_script(script)

    def exported(self):
        return self.js('return window.IMPACTAudioReview.exportReview()')

    def test_autofill_and_no_conclusion_or_speaker_controls(self):
        self.assertEqual(float(self.browser.find_element(By.ID, 'start').get_attribute('value')), .35)
        self.assertEqual(float(self.browser.find_element(By.ID, 'end').get_attribute('value')), .8)
        self.assertEqual(self.js('return document.querySelectorAll("select").length'), 3)
        self.assertEqual(self.exported()['items'], [])

    def test_approvals_and_edits_invalidate_only_changed_field(self):
        self.click('text-ok'); self.click('time-ok')
        self.assertEqual(len(self.exported()['items']), 1)
        self.js('const node=document.getElementById("text");node.value="Corrected text";node.dispatchEvent(new Event("input"));')
        self.assertFalse(self.browser.find_element(By.ID, 'text-ok').is_selected())
        self.assertTrue(self.browser.find_element(By.ID, 'time-ok').is_selected())
        self.click('text-ok')
        self.js('const node=document.getElementById("end");node.value="0.9";node.dispatchEvent(new Event("input"));')
        self.assertTrue(self.browser.find_element(By.ID, 'text-ok').is_selected())
        self.assertFalse(self.browser.find_element(By.ID, 'time-ok').is_selected())
        self.assertEqual(self.exported()['items'], [])

    def test_excluded_and_pending_are_not_in_final_transcript_but_resume(self):
        self.click('text-ok'); self.click('time-ok'); self.click('excluded')
        payload = self.exported()
        self.assertEqual(payload['items'], [])
        self.assertEqual(len(payload['review_records']), 3)
        self.assertTrue(payload['review_records'][0]['excluded'])
        self.browser.refresh()
        self.assertTrue(self.browser.find_element(By.ID, 'excluded').is_selected())
        self.click('excluded')
        self.assertEqual(len(self.exported()['items']), 1)

    def test_invalid_and_empty_boundaries_cannot_be_approved(self):
        for value in ('', '0.2', '8', '-1'):
            self.browser.execute_script('const node=document.getElementById("end");node.value=arguments[0];node.dispatchEvent(new Event("input"));', value)
            self.click('accept')
            self.assertEqual(self.exported()['items'], [])
            self.assertTrue(self.browser.find_element(By.ID, 'validation').text)
        self.browser.execute_script('const node=document.getElementById("end");node.value="";node.dispatchEvent(new Event("input"));')
        self.assertEqual(self.exported()['review_records'][0]['human_end_audio_s'], '')
        self.assertEqual(self.js('''const api=window.IMPACTAudioReview;
          const key='impact-review-'+api.data.language+'-'+api.data.plan_hash;
          return JSON.parse(localStorage.getItem(key)).items.u0.human_end_audio_s;'''), '')
        for _ in range(5):
            self.browser.refresh()
            self.wait_for_page()
            details = self.js('''const api=window.IMPACTAudioReview;
              const key='impact-review-'+api.data.language+'-'+api.data.plan_hash;
              return {plan:api.data.plan_hash, field:document.getElementById('end').value,
                record:api.exportReview().review_records[0], saved:JSON.parse(localStorage.getItem(key))};''')
            self.assertEqual(details['record']['human_end_audio_s'], '', details)
            self.assertEqual(details['field'], '', details)

    def test_changing_recording_cancels_old_clip(self):
        self.click('play')
        self.js('const n=document.getElementById("recording");n.value="two.wav";n.dispatchEvent(new Event("change"));')
        self.assertTrue(self.js('return document.getElementById("player").paused'))
        self.assertAlmostEqual(float(self.browser.find_element(By.ID, 'start').get_attribute('value')), 2.2)
        self.click('play')
        self.wait.until(lambda driver: self.js('return document.getElementById("player").src.includes("two.wav")'))
        self.wait.until(lambda driver: self.js('return document.getElementById("player").paused && document.getElementById("player").currentTime>=2.68'))
        self.assertLessEqual(self.js('return document.getElementById("player").currentTime'), 2.75)

    def test_original_clip_stops_at_end_and_seek_range_is_clip_length(self):
        self.click('original')
        self.wait.until(lambda driver: self.js('return document.getElementById("player").currentTime >= .35'))
        self.wait.until(lambda driver: self.js('return document.getElementById("player").paused && document.getElementById("player").currentTime >= .78'))
        result = self.js('const p=document.getElementById("player");return {time:p.currentTime,max:Number(document.getElementById("seek").max),src:p.src};')
        self.assertLessEqual(result['time'], .85)
        self.assertAlmostEqual(result['max'], .45, places=3)
        self.assertIn('#t=0.35,0.8', result['src'])

    def test_corrected_clip_and_context_have_separate_bounds(self):
        self.js('for (const [id,v] of [["start","1.1"],["end","1.4"]]){const n=document.getElementById(id);n.value=v;n.dispatchEvent(new Event("input"));}')
        self.click('play')
        self.wait.until(lambda driver: self.js('return document.getElementById("player").paused && document.getElementById("player").currentTime>=1.38'))
        self.assertLessEqual(self.js('return document.getElementById("player").currentTime'), 1.45)
        self.click('context')
        self.wait.until(lambda driver: self.js('return document.getElementById("player").src.includes("#t=0,2.8")'))
        self.assertAlmostEqual(self.js('return Number(document.getElementById("seek").max)'), 2.8)

    def test_switching_utterance_cancels_playback_and_keeps_edits(self):
        self.click('play'); self.click('next')
        self.assertTrue(self.js('return document.getElementById("player").paused'))
        self.assertEqual(float(self.browser.find_element(By.ID, 'start').get_attribute('value')), 1.2)
        self.click('play'); self.click('previous')
        time.sleep(.2)
        self.assertTrue(self.js('return document.getElementById("player").paused'))

    def test_storage_failure_is_visible_and_export_keeps_work(self):
        self.js('Storage.prototype.setItem=function(){throw new Error("quota");};')
        self.click('text-ok'); self.click('time-ok')
        self.assertIn('失败', self.browser.find_element(By.ID, 'saved').text)
        # Export the in-memory state through the real download button.
        self.js('window.exportedPayload=null;URL.createObjectURL=function(blob){blob.text().then(t=>window.exportedPayload=JSON.parse(t));return "blob:synthetic"};HTMLAnchorElement.prototype.click=function(){};')
        self.click('export')
        self.wait.until(lambda driver: self.js('return window.exportedPayload!==null'))
        self.assertEqual(self.js('return window.exportedPayload.items.length'), 1)

    def test_legacy_review_migration_and_import_identity_checks(self):
        result = self.js('''
          const C=window.IMPACTAudioReview.core,DATA=window.IMPACTAudioReview.data,i=DATA.items[0],f=DATA.files[0];
          const good=C.normalize(i,{text_decision:'correct_asr',human_text:'Edited',timing_decision:'keep_original'},f);
          const blank=C.normalize(i,{text_decision:'keep_as_spoken',timing_decision:'corrected',human_start_audio_s:'',human_end_audio_s:''},f);
          const other=C.normalize(i,{speaker_role:'experimenter'},f);
          const state={items:{},files:{}};
          const counts=C.importReview(DATA,state,{items:[{review_id:i.review_id,source_text_sha256:'wrong'},
            {...i,...good}]});
          return {good,blank,other,counts,accepted:C.accepted(good,i,f)};''')
        self.assertTrue(result['accepted'])
        self.assertFalse(result['blank']['timing_checked'])
        self.assertEqual(result['blank']['human_start_audio_s'], .35)
        self.assertTrue(result['other']['excluded'])
        self.assertEqual(result['counts'], dict(imported=1, skipped=1))

    def test_actual_import_file_updates_fields_and_preserves_original(self):
        payload = self.exported()
        payload['review_records'][0].update(human_text='Human correction', human_start_audio_s=.4,
                                           human_end_audio_s=.9, text_checked=True, timing_checked=True)
        path = self.folder / 'import.json'
        path.write_text(json.dumps(payload))
        self.browser.find_element(By.ID, 'import-file').send_keys(str(path))
        self.wait.until(lambda driver: driver.find_element(By.ID, 'text').get_attribute('value') == 'Human correction')
        self.assertTrue(self.browser.find_element(By.ID, 'text-ok').is_selected())
        self.assertEqual(len(self.exported()['items']), 1)
        self.assertEqual(self.js('return window.IMPACTAudioReview.data.items[0].original_text'), 'Synthetic utterance one.')

    def test_no_page_scrolling_and_controls_visible_in_three_languages(self):
        for language in ('zh', 'en', 'de'):
            self.browser.get((self.folder / (language + '.html')).as_uri())
            for width, height in ((1366, 768), (1280, 720), (1024, 650), (1024, 600)):
                self.browser.set_window_size(width, height)
                self.assertTrue(self.js('return document.documentElement.scrollHeight <= innerHeight+1'))
                self.assertTrue(self.js('return ["seek","text-ok","time-ok","accept"].every(id=>{let r=document.getElementById(id).getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight;})'))
        self.browser.set_window_size(1366, 768)

    def test_boost_local_audio_is_audible_and_does_not_change_review(self):
        self.js("""
          const proto=(window.AudioContext||window.webkitAudioContext).prototype;
          const original=proto.createGain;
          proto.createGain=function(){const node=original.call(this);window.testGain=node;
            window.testAnalyser=this.createAnalyser();node.connect(window.testAnalyser);return node;};
          document.getElementById('player').muted=false;
          const end=document.getElementById('end');end.value='4.8';end.dispatchEvent(new Event('input'));
        """)
        # Marionette cannot upload a directory. Exercise the same FileList handler
        # using the two files the directory picker supplies in an interactive browser.
        self.js("document.getElementById('audio-folder-files').removeAttribute('webkitdirectory')")
        self.browser.find_element(By.ID, 'audio-folder-files').send_keys(str(self.folder/'one.wav')+'\n'+str(self.folder/'two.wav'))
        self.js("const g=document.getElementById('gain');g.value='4';g.dispatchEvent(new Event('input'))")
        self.click('play')
        try:
            self.wait.until(lambda driver: self.js("return !document.getElementById('player').paused && !!window.testGain"))
        except Exception:
            print('BOOST_DIAGNOSTIC', self.js("return {status:document.getElementById('audio-status').textContent,src:document.getElementById('player').src,error:document.getElementById('player').dataset.playbackError,files:Array.from(document.getElementById('audio-folder-files').files).map(f=>f.name),gain:!!window.testGain,context:window.testGain?.context.state}"))
            raise
        self.assertEqual(self.js('return window.testGain.gain.value'), 4)
        def rms():
            return self.js("const a=new Float32Array(window.testAnalyser.fftSize);window.testAnalyser.getFloatTimeDomainData(a);return Math.sqrt(a.reduce((n,v)=>n+v*v,0)/a.length)")
        self.wait.until(lambda driver: rms() > .03)
        boosted = rms()
        self.js("const g=document.getElementById('gain');g.value='1';g.dispatchEvent(new Event('input'))")
        time.sleep(.25)
        self.assertGreater(boosted, rms()*2)
        self.assertEqual(self.js('return window.IMPACTAudioReview.data.items[0].original_start_audio_s'), .35)
        self.assertEqual(self.exported()['items'], [])
        self.click('play')
        self.js("const r=document.getElementById('recording');r.value='two.wav';r.dispatchEvent(new Event('change'));const e=document.getElementById('end');e.value='5.5';e.dispatchEvent(new Event('input'))")
        self.click('play')
        self.wait.until(lambda driver: self.js("return !document.getElementById('player').paused"))
        self.assertTrue(self.js("return document.getElementById('player').src.startsWith('blob:')"))

    def test_update_downloaded_page_preserves_task_and_saved_review(self):
        import subprocess, sys
        self.click('text-ok'); self.click('time-ok')
        page = self.folder / (self._testMethodName + '.html')
        before = page.read_text(encoding='utf-8')
        page.write_text(before.replace('IMPACT Audio Review', 'Old audio page', 1), encoding='utf-8')
        subprocess.run([sys.executable, str(ROOT/'audio_review/update_pages.py'), str(page)], check=True, capture_output=True)
        self.browser.refresh()
        self.wait_for_page()
        self.assertEqual(len(self.exported()['items']), 1)
        self.assertEqual(self.js('return window.IMPACTAudioReview.data.plan_hash'), self.id())
        self.assertTrue(list((self.folder/'.impact-recovery').glob('review-*.html')))

    def test_boost_without_folder_keeps_native_playback_available(self):
        self.js("const g=document.getElementById('gain');g.value='4';g.dispatchEvent(new Event('input'))")
        self.click('play')
        self.assertTrue(self.browser.find_element(By.ID,'audio-status').text)
        self.js("const g=document.getElementById('gain');g.value='1';g.dispatchEvent(new Event('input'))")
        self.click('play')
        self.wait.until(lambda driver: self.js("return !document.getElementById('player').paused"))

    def test_approve_next_and_keyboard_and_reload_resume(self):
        from selenium.webdriver.common.keys import Keys
        self.browser.find_element(By.ID, 'text').send_keys(Keys.CONTROL, Keys.ENTER)
        self.assertEqual(len(self.exported()['items']), 1)
        self.assertIn('u1', self.browser.find_element(By.ID, 'position').text)
        self.browser.refresh()
        self.assertIn('u1', self.browser.find_element(By.ID, 'position').text)
        self.js('document.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowLeft",altKey:true,bubbles:true}));')
        self.assertIn('u0', self.browser.find_element(By.ID, 'position').text)

    def test_space_plays_when_approval_button_has_focus_without_approving(self):
        from selenium.webdriver.common.keys import Keys
        button = self.browser.find_element(By.ID, 'accept')
        self.js('document.getElementById("accept").focus()')
        button.send_keys(Keys.SPACE)
        self.wait.until(lambda driver: self.js('return document.getElementById("player").src.includes("one.wav")'))
        self.assertEqual(self.exported()['items'], [])
        self.assertIn('u0', self.browser.find_element(By.ID, 'position').text)

    def test_renderer_does_not_replace_or_execute_transcript_content(self):
        data = json.loads(json.dumps(self.data))
        data['plan_hash'] = 'escaped-transcript-rendering'
        data['items'][0]['original_text'] = '</script><img src=x onerror=alert(1)> __CORE__'
        path = self.folder / 'escaped.html'
        path.write_text(renderer.render_page(data, 'en'), encoding='utf-8')
        self.browser.get(path.as_uri())
        self.assertEqual(self.browser.find_element(By.ID, 'text').get_attribute('value'), data['items'][0]['original_text'])
        self.assertEqual(self.js('return document.querySelectorAll("img").length'), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
