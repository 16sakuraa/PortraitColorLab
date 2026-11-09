// The words on both pages and in the brush, in English and Thai, and the switch between
// them. Shared by the web app and the desktop app's page, so it imports nothing.
//
//   t(key, { name: value })   the text for a key in the current language, with {name}s
//                             filled in; English if Thai lacks it, the key if both do
//   translatePage(root)       fills elements marked data-i18n (text), data-i18n-html,
//                             data-i18n-title, data-i18n-alt and data-i18n-aria
//   attachLangSwitch(host)    a TH | EN switch; the choice is remembered in this browser
//   onLangChange(fn)          fn(lang) after each switch
//
// The first visit follows the browser's language. The Thai is written for this app.

const STRINGS = {
  en: {
    'lang.th': 'ภาษาไทย',
    'lang.en': 'English',
    'lang.aria': 'Language',

    'page.tagline': 'Runs in your browser · photos never leave your computer',
    'page.newPhoto': 'New photo',
    'page.download': 'Download',
    'page.chooseAria': 'Choose a photo',
    'page.dropHtml': '<strong>Click to choose a photo</strong> or drag it here',
    'page.dropNoteWeb': 'JPG or PNG, any size. Colour profile and camera data are kept.',
    'page.dropNoteExe': 'JPG, PNG, BMP, WEBP — stays on your computer',
    'page.original': 'Original',
    'page.retouched': 'Retouched',
    'page.altOriginal': 'Original photo',
    'page.altRetouched': 'Retouched photo',
    'page.zoomHint': 'Scroll on either photo to zoom both into the same spot · drag to move around · double-click to see the whole photo',
    'page.settingsAria': 'Retouch settings',
    'page.version': 'version {v}',
    'page.footerHtml': 'Face detection: YuNet (OpenCV Zoo). Skin segmentation and face landmarks: Google MediaPipe selfie multiclass and Face Mesh V2 (Apache 2.0). The optional blemish network (<code>?blemish=network</code>) was trained on <a href="https://github.com/skylab-tech/ffhqr-dataset">FFHQR</a> (CC BY-NC-SA 4.0), for non-commercial use.',

    'panel.skinPhoto': 'Skin and photo',
    'panel.smoothing': 'Smoothing',
    'panel.finishing': 'Finishing',
    'panel.brush': 'Brush',
    'panel.history': 'History',
    'history.undo': 'Undo',
    'history.undoTitle': 'Undo the last change (Ctrl+Z)',
    'history.redo': 'Redo',
    'history.redoTitle': 'Redo what was undone (Ctrl+Y)',
    'history.opened': 'Photo opened',
    'history.cleared': 'All strokes removed',
    'history.on': 'on',
    'history.off': 'off',
    'history.empty': 'Open a photo to start.',
    'history.hint': 'Click a change to go back to it. A new change after that replaces the ones below it.',
    'confirm.newPhoto': 'Start over with another photo? The changes to this one haven\'t been downloaded and will be lost.',
    'ctl.lighting': 'Lighting',
    'ctl.lightingTitle': 'Hard or patchy light (on-camera flash, a bare window) leaves mottled light and shadow; this evens it out more and keeps the face\'s shading. It won\'t blend a sharp shadow edge.',
    'ctl.lightingSoft': 'Soft (studio)',
    'ctl.lightingHard': 'Hard or patchy (flash)',
    'ctl.print': 'Print size',
    'ctl.printTitle': 'A small print keeps coarser grain, so the skin still has texture at that size. It looks coarser here.',
    'ctl.printLarge': 'Large print or screen',
    'ctl.printSmall': 'Small print (1-inch ID)',
    'ctl.faceOnly': 'Face only',
    'ctl.blemish': 'Remove blemishes',
    'ctl.strength': 'Strength',
    'ctl.texture': 'Keep texture',
    'ctl.shine': 'Reduce shine',
    'ctl.shineTitle': 'Tames the bright glints of oily skin. Dark pores, freckles and coloured marks are left alone.',
    'ctl.pores': 'Add pores',
    'note.step1': 'Tinted skin is what gets smoothed. Paint the skin area to add skin the mask missed, or take out what shouldn\'t be smoothed.',
    'note.step2': 'The light brush evens out blotches, dark circles and seams between light and shadow with a colour you pick from the skin, Nose line paints a soft highlight down the nose, and Smooth again smooths leftover coarse grain or stubble shadow once more.',
    'note.step3': 'Bring back sharpness around the eyes, nostrils and jawline with Undo smoothing, cover scars with Heal, and click leftover spots away with Remove spot. Then press Download.',

    'step1.title': 'Skin',
    'step1.button': 'Paint the skin area',
    'step1.about': 'Choose the skin to smooth, the lighting and the print size',
    'step2.title': 'Smooth',
    'step2.button': 'Use the smoothing brushes',
    'step2.about': 'Remove blemishes, smooth, and even out light and shadow',
    'step3.title': 'Finish',
    'step3.button': 'Use the finishing brushes',
    'step3.about': 'Bring back sharpness, heal scars, remove leftover spots',
    'steps.back': 'Back: {step}',
    'steps.next': 'Next: {step}',

    'brush.use': 'Use brush',
    'brush.done': 'Done with brush',
    'brush.strength': 'Strength',
    'brush.size': 'Size',
    'brush.sizeTitle': 'Brush size ( [ and ] )',
    'brush.softness': 'Softness',
    'brush.softTitle': 'How soft the brush\'s edge is',
    'brush.undo': 'Undo',
    'brush.undoTitle': 'Undo the last stroke (Ctrl+Z)',
    'brush.reset': 'Reset',
    'brush.resetTitle': 'Remove all strokes',
    'brush.fit': 'Fit',
    'brush.fitTitle': 'Show the whole photo (0)',
    'brush.tint': 'Show tint',
    'brush.tintTitle': 'Show the tint (M)',
    'brush.aligned': 'Aligned',
    'brush.alignedTitle': 'On: the source keeps its distance from the brush from one stroke to the next. Off: every stroke starts copying from the point you picked.',
    'brush.swatch': 'The picked colour',
    'brush.move': ' Scroll to zoom, hold Space or the right mouse button to move around.',

    'mode.protect.label': 'Restore',
    'mode.protect.title': 'Bring back the original here: at 100% nothing is smoothed, no spots are filled, and earlier heals and light are taken back (X switches)',
    'mode.protect.strength': 'How much of the original to bring back',
    'mode.protect.hint': 'Tinted skin gets smoothed. Restore brings back the original as it was, for example on a mole: no smoothing, no spot fills, and heals and light brush painted before it are taken back. Turn the tint off (M) to see the result. Where strokes cross, the later one wins: 40% over 100% leaves 40%.',
    'mode.removeSkin.label': 'Remove skin',
    'mode.removeSkin.title': 'Take this area out of the smoothing: no smoothing and no spot fills here (X switches)',
    'mode.removeSkin.strength': 'How much to take out',
    'mode.removeSkin.hint': 'Tinted skin gets smoothed. Add skin brings an area in; Remove skin takes one out, with no smoothing and no spot fills, for example eyes, nostrils or mouth corners that got tinted. Turn the tint off (M) to see the result.',
    'mode.unsmooth.label': 'Undo smoothing',
    'mode.unsmooth.title': 'Bring back the original\'s sharpness and texture here, but keep removed spots, heals and light (X switches)',
    'mode.unsmooth.strength': 'How much of the original sharpness to bring back',
    'mode.unsmooth.hint': 'Tinted skin gets smoothed. Undo smoothing brings back the original\'s sharpness and texture, for example around the eyes, the nostrils or the jawline, and keeps the removed spots, heals and light brush. Turn the tint off (M) to see the result. Where strokes cross, the later one wins.',
    'mode.skin.label': 'Add skin',
    'mode.skin.title': 'Smooth here, even if the automatic mask left it out (X switches)',
    'mode.skin.strength': 'How much to smooth',
    'mode.skin.hint': 'Tinted skin gets smoothed. Add skin smooths an area the automatic mask left out, but doesn\'t look for spots in it. Where strokes cross, the later one wins.',
    'mode.heal.label': 'Heal',
    'mode.heal.title': 'Paint over a scar, cut or mark with skin copied from elsewhere, matched to the light here. Alt-click to pick where it copies from (X switches)',
    'mode.heal.strength': 'How much of the copy to paint',
    'mode.heal.hint': 'Alt-click clean skin to copy from (or press Set source), then paint over a scar, cut or large spot. The skin\'s detail is copied and its light and shading are matched to where you paint. Make the brush just cover the mark.',
    'mode.heal.point': 'Set source',
    'mode.heal.pointTitle': 'Then click clean skin to copy from. Alt-click does the same at any time.',
    'mode.heal.noPoint': 'Pick the skin to copy from first: Alt-click it, or press Set source and click it.',
    'mode.spot.label': 'Remove spot',
    'mode.spot.title': 'Click a spot to fill it from the skin around it (X switches)',
    'mode.spot.strength': 'How much of the fill to paint',
    'mode.spot.hint': 'Click a spot that wasn\'t removed by itself: it\'s filled from the skin around it. Make the brush a little bigger than the spot.',
    'mode.light.label': 'Light brush',
    'mode.light.title': 'Paint a colour picked from the skin over light and shadow, keeping the skin\'s grain. Alt-click to pick the colour (X switches)',
    'mode.light.strength': 'How much of the picked colour to paint: low, around 20%, and build it up',
    'mode.light.hint': 'Alt-click the skin whose colour you want (or press Pick colour), then paint at a low Strength over blotches, seams between light and shadow, frown lines or dark circles. Only the light and colour change; the skin\'s grain stays. Judge it from a distance.',
    'mode.light.point': 'Pick colour',
    'mode.light.pointTitle': 'Then click the skin whose colour you want. Alt-click does the same at any time.',
    'mode.light.noPoint': 'Pick the colour first: Alt-click the skin to take it from, or press Pick colour and click it.',
    'mode.line.label': 'Nose line',
    'mode.line.title': 'Click twice to paint a soft line of the picked colour between the clicks, such as a highlight down the nose bridge (X switches)',
    'mode.line.strength': 'How much of the picked colour to paint: around 20-30%',
    'mode.line.hint': 'Alt-click the brightest skin on the nose to pick the highlight colour (or press Pick colour; it\'s shared with the light brush). Then click at the top of the nose bridge and at the tip: a soft line of that colour is painted between them, keeping the skin\'s grain. Make the brush as wide as the bridge.',
    'mode.line.next': 'Now click the other end of the line. Esc cancels.',
    'mode.line.point': 'Pick colour',
    'mode.line.pointTitle': 'Then click the skin whose colour you want. Alt-click does the same at any time.',
    'mode.line.noPoint': 'Pick the colour first: Alt-click the brightest skin on the nose, or press Pick colour and click it.',
    'mode.resmooth.label': 'Smooth again',
    'mode.resmooth.title': 'Smooth here once more, with a finer grain: for coarse grain, blotches or stubble shadow that are left (X switches)',
    'mode.resmooth.strength': 'How much of the finer smoothing to use',
    'mode.resmooth.hint': 'Paint over skin that still looks coarse or blotchy, or over stubble shadow: it\'s smoothed again at full strength with half the grain, so only the finest grain stays. It works even where the tint left skin out.',

    'status.loading': 'Loading (about 60 MB the first time, then cached)…',
    'status.ready': 'Ready · running on the {engine}.',
    'engine.gpu': 'graphics chip (WebGPU)',
    'engine.cpu': 'CPU — slower; a browser with WebGPU (Chrome or Edge) is faster',
    'status.stillLoading': 'Still loading, one moment…',
    'status.wrong': 'Something went wrong: {message}',
    'status.processing': 'Processing…',
    'status.error': 'Error: {message}',
    'status.noFaceModel': ' — skin-tone mode (no face model installed)',
    'status.modelMissing': 'model missing',
    'summary.faces1': '1 face found',
    'summary.faces': '{n} faces found',
    'summary.spots1': '1 spot removed',
    'summary.spots': '{n} spots removed',
    'summary.healed1': 'blemishes treated on 1 face',
    'summary.healed': 'blemishes treated on {n} faces',
    'summary.profile': 'colour profile kept',
    'summary.time': 'analysed in {s} s',
    'summary.saved': 'saved {name} ({mb} MB)',
    'progress.tools': 'Loading image tools…',
    'progress.models': 'Loading face models…',
    'progress.reading': 'Reading photo…',
    'progress.finding': 'Finding faces, skin and blemishes…',
    'progress.mask': 'Updating the skin mask…',
    'progress.full': 'Processing full resolution…',
    'progress.saving': 'Saving…',

    'update.downloading': 'Downloading version {v}...',
    'update.ready': 'Version {v} is ready. To use it, close the Skin Smoother console window and open the program again.',
    'update.failed': 'Couldn\'t update to version {v}: {reason}. It will try again next time.',
  },

  th: {
    'lang.aria': 'ภาษา',

    'page.tagline': 'ทำงานในเบราว์เซอร์ · รูปไม่ถูกส่งออกจากเครื่องของคุณ',
    'page.newPhoto': 'รูปใหม่',
    'page.download': 'ดาวน์โหลด',
    'page.chooseAria': 'เลือกรูป',
    'page.dropHtml': '<strong>คลิกเพื่อเลือกรูป</strong> หรือลากรูปมาวางที่นี่',
    'page.dropNoteWeb': 'JPG หรือ PNG ขนาดเท่าไรก็ได้ โปรไฟล์สีและข้อมูลกล้องยังอยู่ครบ',
    'page.dropNoteExe': 'JPG, PNG, BMP, WEBP — รูปอยู่ในเครื่องของคุณเท่านั้น',
    'page.original': 'ต้นฉบับ',
    'page.retouched': 'แต่งแล้ว',
    'page.altOriginal': 'รูปต้นฉบับ',
    'page.altRetouched': 'รูปที่แต่งแล้ว',
    'page.zoomHint': 'หมุนลูกกลิ้งบนรูปใดก็ได้เพื่อซูมทั้งสองรูปเข้าไปที่จุดเดียวกัน · ลากเพื่อเลื่อนดู · ดับเบิลคลิกเพื่อดูทั้งรูป',
    'page.settingsAria': 'การตั้งค่าการแต่งภาพ',
    'page.version': 'เวอร์ชัน {v}',
    'page.footerHtml': 'ตรวจหาใบหน้า: YuNet (OpenCV Zoo) แยกผิวและหาตำแหน่งบนใบหน้า: Google MediaPipe selfie multiclass และ Face Mesh V2 (Apache 2.0) โมเดลลบสิวแบบเลือกใช้ (<code>?blemish=network</code>) ฝึกจาก <a href="https://github.com/skylab-tech/ffhqr-dataset">FFHQR</a> (CC BY-NC-SA 4.0) ใช้ได้เฉพาะงานที่ไม่ใช่เชิงพาณิชย์',

    'panel.skinPhoto': 'ผิวและรูปถ่าย',
    'panel.smoothing': 'เกลี่ยผิว',
    'panel.finishing': 'เก็บงาน',
    'panel.brush': 'บรัช',
    'panel.history': 'ประวัติการแต่ง',
    'history.undo': 'เลิกทำ',
    'history.undoTitle': 'ย้อนการแก้ไขล่าสุด (Ctrl+Z)',
    'history.redo': 'ทำซ้ำ',
    'history.redoTitle': 'ทำสิ่งที่ย้อนไปกลับมาอีกครั้ง (Ctrl+Y)',
    'history.opened': 'เปิดรูป',
    'history.cleared': 'ล้างเส้นที่ระบายทั้งหมด',
    'history.on': 'เปิด',
    'history.off': 'ปิด',
    'history.empty': 'เปิดรูปเพื่อเริ่ม',
    'confirm.newPhoto': 'เริ่มรูปใหม่ใช่ไหม? การแก้ไขรูปนี้ยังไม่ได้ดาวน์โหลด และจะหายไป',
    'history.hint': 'คลิกรายการใดก็ได้เพื่อย้อนกลับไปตอนนั้น ถ้าแก้ไขต่อจากตรงนั้น รายการที่อยู่ถัดลงไปจะถูกแทนที่',
    'ctl.lighting': 'แสงที่ใช้ถ่าย',
    'ctl.lightingTitle': 'แสงแข็งหรือแสงไม่สม่ำเสมอ (แฟลชบนกล้อง แสงจากหน้าต่าง) ทำให้ผิวมีแสงเงาเป็นด่าง ตัวเลือกนี้เกลี่ยให้เรียบขึ้นโดยยังคงแสงเงาของใบหน้าไว้ แต่จะไม่เกลี่ยขอบเงาที่คม',
    'ctl.lightingSoft': 'แสงนุ่ม (สตูดิโอ)',
    'ctl.lightingHard': 'แสงแข็งหรือไม่สม่ำเสมอ (แฟลช)',
    'ctl.print': 'ขนาดที่จะพิมพ์',
    'ctl.printTitle': 'รูปเล็กจะเก็บเม็ดผิวที่หยาบกว่าไว้ เพื่อให้ผิวยังมีรายละเอียดเมื่อพิมพ์ขนาดนั้น บนจอจึงดูหยาบกว่า',
    'ctl.printLarge': 'รูปใหญ่หรือดูบนจอ',
    'ctl.printSmall': 'รูปเล็ก (ติดบัตร 1 นิ้ว)',
    'ctl.faceOnly': 'เฉพาะใบหน้า',
    'ctl.blemish': 'ลบสิวและจุดด่าง',
    'ctl.strength': 'ความแรง',
    'ctl.texture': 'คงรายละเอียดผิว',
    'ctl.shine': 'ลดความมัน',
    'ctl.shineTitle': 'ลดจุดสะท้อนแสงบนผิวมัน รูขุมขนสีเข้ม กระ และรอยที่มีสีจะไม่ถูกแตะ',
    'ctl.pores': 'เพิ่มรูขุมขน',
    'note.step1': 'ส่วนที่มีสีทับคือผิวที่จะถูกแต่ง ใช้บรัชระบายเพื่อเพิ่มผิวที่ระบบเลือกไม่ครบ หรือเอาส่วนที่ไม่ควรแต่งออก',
    'note.step2': 'บรัชเกลี่ยแสงช่วยเกลี่ยรอยด่าง รอยคล้ำใต้ตา และรอยต่อระหว่างแสงกับเงา ด้วยสีที่ดูดจากผิว เส้นสันจมูกลากไฮไลต์อ่อน ๆ ตามสันจมูก และเกลี่ยผิวซ้ำช่วยเกลี่ยเม็ดผิวหยาบหรือเงาหนวดเคราที่ยังเหลืออีกรอบ',
    'note.step3': 'เรียกความคมกลับรอบดวงตา ปีกจมูก และกรอบหน้าด้วย “เรียกความคมกลับ” ปิดแผลเป็นด้วย “ซ่อมผิว” และคลิกลบจุดที่ยังเหลือด้วย “ลบสิว” จากนั้นกดดาวน์โหลด',

    'step1.title': 'ผิว',
    'step1.button': 'ระบายพื้นที่ผิว',
    'step1.about': 'เลือกผิวที่จะแต่ง แสงที่ใช้ถ่าย และขนาดที่จะพิมพ์',
    'step2.title': 'เกลี่ยผิว',
    'step2.button': 'ใช้บรัชเกลี่ยผิว',
    'step2.about': 'ลบสิว เกลี่ยผิว และเกลี่ยแสงเงา',
    'step3.title': 'เก็บงาน',
    'step3.button': 'ใช้บรัชเก็บงาน',
    'step3.about': 'เรียกความคมกลับ ซ่อมแผลเป็น ลบจุดที่ยังเหลือ',
    'steps.back': 'ย้อนกลับ: {step}',
    'steps.next': 'ถัดไป: {step}',

    'brush.use': 'ใช้บรัช',
    'brush.done': 'ใช้บรัชเสร็จแล้ว',
    'brush.strength': 'ความแรง',
    'brush.size': 'ขนาด',
    'brush.sizeTitle': 'ขนาดบรัช ( [ และ ] )',
    'brush.softness': 'ขอบนุ่ม',
    'brush.softTitle': 'ขอบบรัชนุ่มแค่ไหน',
    'brush.undo': 'เลิกทำ',
    'brush.undoTitle': 'ยกเลิกเส้นล่าสุด (Ctrl+Z)',
    'brush.reset': 'ล้างทั้งหมด',
    'brush.resetTitle': 'ลบทุกเส้นที่ระบายไว้',
    'brush.fit': 'พอดีจอ',
    'brush.fitTitle': 'แสดงทั้งรูป (0)',
    'brush.tint': 'แสดงสีทับ',
    'brush.tintTitle': 'แสดงสีทับ (M)',
    'brush.aligned': 'ตามบรัช',
    'brush.alignedTitle': 'เปิด: จุดต้นแบบเลื่อนตามบรัชโดยรักษาระยะห่างเดิมทุกเส้น ปิด: ทุกเส้นเริ่มคัดลอกจากจุดที่เลือกไว้',
    'brush.swatch': 'สีที่ดูดไว้',
    'brush.move': ' หมุนลูกกลิ้งเพื่อซูม กด Space ค้างหรือกดปุ่มขวาของเมาส์ค้างเพื่อเลื่อนดูรูป',

    'mode.protect.label': 'คืนภาพเดิม',
    'mode.protect.title': 'คืนภาพต้นฉบับตรงนี้: ที่ 100% จะไม่เกลี่ยผิว ไม่ลบสิว และยกเลิกการซ่อมผิวและแสงที่ระบายไว้ก่อนหน้า (X สลับ)',
    'mode.protect.strength': 'คืนภาพเดิมกลับมามากแค่ไหน',
    'mode.protect.hint': 'ส่วนที่มีสีทับคือผิวที่จะถูกแต่ง “คืนภาพเดิม” นำต้นฉบับกลับมาตามเดิม เช่น ตรงไฝ: ไม่เกลี่ยผิว ไม่ลบสิว และยกเลิกการซ่อมผิวกับบรัชเกลี่ยแสงที่ระบายไว้ก่อนหน้า ปิดสีทับ (M) เพื่อดูผล ตรงที่เส้นทับกัน เส้นที่ระบายทีหลังมีผล: 40% ทับ 100% จะเหลือ 40%',
    'mode.removeSkin.label': 'เอาผิวออก',
    'mode.removeSkin.title': 'เอาพื้นที่นี้ออกจากการแต่ง: ไม่เกลี่ยผิวและไม่ลบสิวตรงนี้ (X สลับ)',
    'mode.removeSkin.strength': 'เอาออกมากแค่ไหน',
    'mode.removeSkin.hint': 'ส่วนที่มีสีทับคือผิวที่จะถูกแต่ง “เพิ่มผิว” เพิ่มพื้นที่เข้าไป “เอาผิวออก” เอาพื้นที่ออก โดยไม่เกลี่ยผิวและไม่ลบสิว เช่น ตา รูจมูก หรือมุมปากที่ถูกเลือกติดมา ปิดสีทับ (M) เพื่อดูผล',
    'mode.unsmooth.label': 'เรียกความคมกลับ',
    'mode.unsmooth.title': 'นำความคมและรายละเอียดผิวของต้นฉบับกลับมาตรงนี้ โดยยังเก็บการลบสิว การซ่อมผิว และแสงไว้ (X สลับ)',
    'mode.unsmooth.strength': 'เรียกความคมของต้นฉบับกลับมามากแค่ไหน',
    'mode.unsmooth.hint': 'ส่วนที่มีสีทับคือผิวที่จะถูกแต่ง “เรียกความคมกลับ” นำความคมและรายละเอียดผิวของต้นฉบับกลับมา เช่น รอบดวงตา ปีกจมูก หรือกรอบหน้า โดยยังเก็บการลบสิว การซ่อมผิว และบรัชเกลี่ยแสงไว้ ปิดสีทับ (M) เพื่อดูผล ตรงที่เส้นทับกัน เส้นที่ระบายทีหลังมีผล',
    'mode.skin.label': 'เพิ่มผิว',
    'mode.skin.title': 'เกลี่ยผิวตรงนี้ แม้ระบบจะไม่ได้เลือกไว้ (X สลับ)',
    'mode.skin.strength': 'เกลี่ยมากแค่ไหน',
    'mode.skin.hint': 'ส่วนที่มีสีทับคือผิวที่จะถูกแต่ง “เพิ่มผิว” เกลี่ยพื้นที่ที่ระบบเลือกไม่ครบ แต่จะไม่หาสิวในบริเวณนั้น ตรงที่เส้นทับกัน เส้นที่ระบายทีหลังมีผล',
    'mode.heal.label': 'ซ่อมผิว',
    'mode.heal.title': 'ระบายทับแผลเป็น รอยแผล หรือรอยด่าง ด้วยผิวที่คัดลอกจากที่อื่น โดยปรับแสงให้เข้ากับตรงนี้ Alt+คลิกเพื่อเลือกจุดต้นแบบ (X สลับ)',
    'mode.heal.strength': 'ระบายผิวที่คัดลอกมามากแค่ไหน',
    'mode.heal.hint': 'Alt+คลิกบนผิวที่เรียบเพื่อใช้เป็นต้นแบบ (หรือกด “ตั้งต้นแบบ”) แล้วระบายทับแผลเป็น รอยแผล หรือสิวเม็ดใหญ่ รายละเอียดผิวจะถูกคัดลอกมา และแสงเงาจะถูกปรับให้เข้ากับตรงที่ระบาย ตั้งขนาดบรัชให้คลุมรอยพอดี',
    'mode.heal.point': 'ตั้งต้นแบบ',
    'mode.heal.pointTitle': 'แล้วคลิกผิวที่เรียบเพื่อใช้เป็นต้นแบบ Alt+คลิกก็ได้ตลอดเวลา',
    'mode.heal.noPoint': 'เลือกผิวต้นแบบก่อน: Alt+คลิกที่ผิว หรือกด “ตั้งต้นแบบ” แล้วคลิก',
    'mode.spot.label': 'ลบสิว',
    'mode.spot.title': 'คลิกที่สิวหรือจุดด่างเพื่อเติมด้วยผิวรอบ ๆ (X สลับ)',
    'mode.spot.strength': 'เติมมากแค่ไหน',
    'mode.spot.hint': 'คลิกที่สิวหรือจุดที่ระบบไม่ได้ลบให้ จุดนั้นจะถูกเติมด้วยผิวรอบ ๆ ตั้งขนาดบรัชให้ใหญ่กว่าจุดเล็กน้อย',
    'mode.light.label': 'บรัชเกลี่ยแสง',
    'mode.light.title': 'ระบายสีที่ดูดจากผิวทับแสงเงา โดยยังคงเม็ดผิวไว้ Alt+คลิกเพื่อดูดสี (X สลับ)',
    'mode.light.strength': 'ระบายสีที่ดูดมามากแค่ไหน: ใช้น้อย ราว 20% แล้วค่อย ๆ เพิ่ม',
    'mode.light.hint': 'Alt+คลิกที่ผิวสีที่ต้องการ (หรือกด “ดูดสี”) แล้วระบายเบา ๆ ทับรอยด่าง รอยต่อระหว่างแสงกับเงา รอยย่นระหว่างคิ้ว หรือรอยคล้ำใต้ตา จะเปลี่ยนเฉพาะแสงและสี เม็ดผิวยังอยู่ ควรดูผลจากระยะไกล',
    'mode.light.point': 'ดูดสี',
    'mode.light.pointTitle': 'แล้วคลิกผิวสีที่ต้องการ Alt+คลิกก็ได้ตลอดเวลา',
    'mode.light.noPoint': 'ดูดสีก่อน: Alt+คลิกที่ผิว หรือกด “ดูดสี” แล้วคลิก',
    'mode.line.label': 'เส้นสันจมูก',
    'mode.line.title': 'คลิกสองครั้งเพื่อลากเส้นนุ่ม ๆ ด้วยสีที่ดูดไว้ระหว่างสองจุด เช่น ไฮไลต์ตามสันจมูก (X สลับ)',
    'mode.line.strength': 'ระบายสีที่ดูดมามากแค่ไหน: ราว 20-30%',
    'mode.line.hint': 'Alt+คลิกที่ผิวส่วนที่สว่างที่สุดบนจมูกเพื่อดูดสีไฮไลต์ (หรือกด “ดูดสี” ใช้สีเดียวกับบรัชเกลี่ยแสง) แล้วคลิกที่ต้นสันจมูกและที่ปลายจมูก จะได้เส้นสีนั้นแบบนุ่ม ๆ ระหว่างสองจุด โดยยังคงเม็ดผิวไว้ ตั้งขนาดบรัชให้กว้างเท่าสันจมูก',
    'mode.line.next': 'คลิกที่ปลายอีกด้านของเส้น กด Esc เพื่อยกเลิก',
    'mode.line.point': 'ดูดสี',
    'mode.line.pointTitle': 'แล้วคลิกผิวสีที่ต้องการ Alt+คลิกก็ได้ตลอดเวลา',
    'mode.line.noPoint': 'ดูดสีก่อน: Alt+คลิกที่ผิวส่วนที่สว่างที่สุดบนจมูก หรือกด “ดูดสี” แล้วคลิก',
    'mode.resmooth.label': 'เกลี่ยผิวซ้ำ',
    'mode.resmooth.title': 'เกลี่ยผิวตรงนี้อีกรอบด้วยเม็ดผิวที่ละเอียดกว่า สำหรับเม็ดผิวหยาบ รอยด่าง หรือเงาหนวดเคราที่ยังเหลือ (X สลับ)',
    'mode.resmooth.strength': 'ใช้การเกลี่ยซ้ำมากแค่ไหน',
    'mode.resmooth.hint': 'ระบายทับผิวที่ยังดูหยาบหรือด่าง หรือเงาหนวดเครา จะถูกเกลี่ยซ้ำเต็มแรงด้วยเม็ดผิวครึ่งหนึ่ง เหลือไว้เฉพาะเม็ดผิวที่ละเอียดที่สุด ใช้ได้แม้ตรงที่ไม่มีสีทับ',

    'status.loading': 'กำลังโหลด (ครั้งแรกประมาณ 60 MB หลังจากนั้นเก็บไว้ในเครื่อง)…',
    'status.ready': 'พร้อมใช้งาน · ทำงานด้วย {engine}',
    'engine.gpu': 'ชิปกราฟิก (WebGPU)',
    'engine.cpu': 'CPU — ช้ากว่า เบราว์เซอร์ที่มี WebGPU (Chrome หรือ Edge) จะเร็วกว่า',
    'status.stillLoading': 'ยังโหลดอยู่ รอสักครู่…',
    'status.wrong': 'เกิดข้อผิดพลาด: {message}',
    'status.processing': 'กำลังแต่งภาพ…',
    'status.error': 'ข้อผิดพลาด: {message}',
    'status.noFaceModel': ' — โหมดสีผิว (ไม่ได้ติดตั้งโมเดลใบหน้า)',
    'status.modelMissing': 'ไม่มีโมเดล',
    'summary.faces1': 'พบ 1 ใบหน้า',
    'summary.faces': 'พบ {n} ใบหน้า',
    'summary.spots1': 'ลบสิว 1 จุด',
    'summary.spots': 'ลบสิว {n} จุด',
    'summary.healed1': 'แต่งสิวบน 1 ใบหน้า',
    'summary.healed': 'แต่งสิวบน {n} ใบหน้า',
    'summary.profile': 'เก็บโปรไฟล์สีไว้',
    'summary.time': 'วิเคราะห์ใน {s} วินาที',
    'summary.saved': 'บันทึก {name} ({mb} MB) แล้ว',
    'progress.tools': 'กำลังโหลดเครื่องมือแต่งภาพ…',
    'progress.models': 'กำลังโหลดโมเดลใบหน้า…',
    'progress.reading': 'กำลังอ่านรูป…',
    'progress.finding': 'กำลังหาใบหน้า ผิว และสิว…',
    'progress.mask': 'กำลังอัปเดตพื้นที่ผิว…',
    'progress.full': 'กำลังแต่งภาพขนาดเต็ม…',
    'progress.saving': 'กำลังบันทึก…',

    'update.downloading': 'กำลังดาวน์โหลดเวอร์ชัน {v}...',
    'update.ready': 'เวอร์ชัน {v} พร้อมแล้ว ปิดหน้าต่างคอนโซลของ Skin Smoother แล้วเปิดโปรแกรมใหม่เพื่อใช้งาน',
    'update.failed': 'อัปเดตเป็นเวอร์ชัน {v} ไม่สำเร็จ ({reason}) จะลองใหม่ในครั้งหน้า',
  },
};

const STORE = 'skinsmoother-lang';
const STYLE = `
  .lang-switch { display: inline-flex; border-radius: 8px; overflow: hidden; align-self: center; }
  .lang-switch button { border-radius: 0; padding: 6px 10px; font-size: 12px; background: #6b7280; color: #fff; }
  .lang-switch button.on { background: #4b7bec; }
`;

function firstLang() {
  try {
    const saved = localStorage.getItem(STORE);
    if (saved && STRINGS[saved]) return saved;
  } catch { /* no storage: follow the browser */ }
  return (navigator.language || '').toLowerCase().startsWith('th') ? 'th' : 'en';
}

let lang = firstLang();
const listeners = new Set();

export const LANGS = Object.keys(STRINGS);
export const getLang = () => lang;
export const has = (key) => key in STRINGS.en;

export function t(key, vars = {}) {
  const text = STRINGS[lang][key] ?? STRINGS.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (whole, name) => (name in vars ? String(vars[name]) : whole));
}

export function translatePage(root = document) {
  for (const node of root.querySelectorAll('[data-i18n]')) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll('[data-i18n-html]')) node.innerHTML = t(node.dataset.i18nHtml);
  for (const node of root.querySelectorAll('[data-i18n-title]')) node.title = t(node.dataset.i18nTitle);
  for (const node of root.querySelectorAll('[data-i18n-alt]')) node.alt = t(node.dataset.i18nAlt);
  for (const node of root.querySelectorAll('[data-i18n-aria]')) node.setAttribute('aria-label', t(node.dataset.i18nAria));
  if (root === document) document.documentElement.lang = lang;
}

export function setLang(next) {
  if (!STRINGS[next] || next === lang) return;
  lang = next;
  try { localStorage.setItem(STORE, next); } catch { /* remembered for this page only */ }
  translatePage();
  for (const fn of listeners) fn(lang);
}

export function onLangChange(fn) { listeners.add(fn); }

// A TH | EN switch, added to host.
export function attachLangSwitch(host) {
  if (!document.getElementById('lang-switch-style')) {
    document.head.append(Object.assign(document.createElement('style'), { id: 'lang-switch-style', textContent: STYLE }));
  }
  const box = Object.assign(document.createElement('span'), { className: 'lang-switch', role: 'group' });
  box.dataset.i18nAria = 'lang.aria';
  const buttons = ['th', 'en'].map((code) => {
    const button = Object.assign(document.createElement('button'), { type: 'button', textContent: code.toUpperCase() });
    button.lang = code;
    button.title = STRINGS.en[`lang.${code}`];
    button.addEventListener('click', () => setLang(code));
    return button;
  });
  const mark = () => buttons.forEach((b) => b.classList.toggle('on', b.lang === lang));
  box.append(...buttons);
  host.prepend(box);
  box.setAttribute('aria-label', t('lang.aria'));
  mark();
  onLangChange(mark);
}
