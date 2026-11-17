// Looks: several settings chosen together. Natural is how the app opens. Studio ID photo
// follows how a portrait studio retouches ID photos by hand: fitted to a set of its
// before/after pairs, it evens blotches and folds more (the wider light setting), keeps a
// small print's grain, and balances skin tone. Under-eyes stays off in it: with the wider
// light setting the skin under the eyes already comes out as it does by hand, and any
// more moved it further from the hand retouch; the slider is for heavy bags.
// Shared by the web app and the desktop app's page, which use the same ids for their
// controls. The desktop app's own window keeps the same values (LOOKS in look.py).

export const LOOKS = {
  natural: { strength: '60', texture: '60', lighting: 'soft', print: 'large', eyes: '0', tone: '0' },
  studio: { strength: '80', texture: '75', lighting: 'hard', print: 'small', eyes: '0', tone: '100' },
};

const $ = (id) => document.getElementById(id);

// The look the controls now show, or 'custom'.
export function lookOf(read = (id) => $(id).value) {
  for (const [name, values] of Object.entries(LOOKS)) {
    if (Object.entries(values).every(([id, value]) => read(id) === value)) return name;
  }
  return 'custom';
}

// select: the Look <select> (options natural, studio and custom). onApply: called once a
// look's settings are in the controls, to show the photo with them. Choosing Custom
// changes nothing; moving a setting a look sets shows the look the controls then match.
// Attach this before the History panel, so its entry for a look holds the look's settings.
export function attachLook({ select, onApply }) {
  const sync = () => { select.value = lookOf(); };
  select.addEventListener('change', () => {
    const values = LOOKS[select.value];
    if (!values) { sync(); return; }
    for (const [id, value] of Object.entries(values)) {
      const input = $(id);
      input.value = value;
      if (input.type === 'range') input.dispatchEvent(new Event('input'));   // the number beside it follows
    }
    onApply();
  });
  for (const id of Object.keys(LOOKS.natural)) $(id).addEventListener('change', sync);
  sync();
  return { sync };
}
