/**
 * Presentation runtime (no dependencies). See docs/PRESENTATION_SYSTEM.md.
 *
 * Markup contract:
 *   <div data-deck>
 *     <div class="deck__stage"> <section class="slide"> ... <aside class="notes">…</aside></section> ... </div>
 *   </div>
 *   Fragments: any element with [data-fragment] (optional value = order index).
 *
 * Keys: → ↓ Space PgDn = next · ← ↑ PgUp = previous · Home/End · F fullscreen ·
 *       O overview · S speaker view · P print · ? help · digits + Enter = go to slide.
 * URL: #/<slide>/<fragment> (1-based slide). ?print renders all slides for PDF.
 * Events dispatched on each slide: 'slide:enter' and 'slide:leave'.
 */

const W = 1600, H = 900;

export function initDeck(root: HTMLElement) {
  const stage = root.querySelector<HTMLElement>('.deck__stage')!;
  const slides = [...stage.querySelectorAll<HTMLElement>(':scope > .slide')];
  const params = new URLSearchParams(location.search);
  const isSpeaker = params.has('speaker');
  const isPrint = params.has('print');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const channel = 'BroadcastChannel' in window ? new BroadcastChannel(`deck:${location.pathname}`) : null;

  slides.forEach((s, i) => {
    s.setAttribute('aria-roledescription', 'slide');
    s.setAttribute('aria-label', `${i + 1} of ${slides.length}`);
    s.dataset.index = String(i);
    s.dataset.num = String(i + 1);
  });
  const fragmentsOf = (s: HTMLElement) =>
    [...s.querySelectorAll<HTMLElement>('[data-fragment]')].sort((a, b) => (+(a.dataset.fragment || 0)) - (+(b.dataset.fragment || 0)));

  if (isPrint) {
    root.classList.add('is-print');
    slides.forEach((s) => { s.classList.add('is-active'); fragmentsOf(s).forEach((f) => f.classList.add('is-visible')); });
    document.querySelectorAll<HTMLVideoElement>('video').forEach((v) => (v.preload = 'metadata'));
    setTimeout(() => window.print(), 600);
    return;
  }

  let cur = -1, frag = 0;
  const counter = root.querySelector<HTMLElement>('[data-counter]');
  const bar = root.querySelector<HTMLElement>('[data-progress]');
  const live = root.querySelector<HTMLElement>('[data-live]');

  const fit = () => {
    const box = root.getBoundingClientRect();
    const scale = Math.min(box.width / W, box.height / H);
    stage.style.transform = `translate(-50%, -50%) scale(${scale})`;
  };
  new ResizeObserver(fit).observe(root);
  fit();

  const show = (i: number, f = 0, broadcast = true) => {
    i = Math.max(0, Math.min(slides.length - 1, i));
    const frs = fragmentsOf(slides[i]);
    f = Math.max(0, Math.min(frs.length, f));
    if (i !== cur) {
      if (cur >= 0) {
        slides[cur].classList.remove('is-active');
        slides[cur].setAttribute('aria-hidden', 'true');
        slides[cur].inert = true;
        slides[cur].dispatchEvent(new CustomEvent('slide:leave'));
      }
      slides[i].classList.add('is-active');
      slides[i].removeAttribute('aria-hidden');
      slides[i].inert = false;
      slides[i].dispatchEvent(new CustomEvent('slide:enter'));
      if (live) live.textContent = `Slide ${i + 1} of ${slides.length}: ${slides[i].dataset.title ?? ''}`;
    }
    frs.forEach((el, k) => el.classList.toggle('is-visible', k < f));
    frs.forEach((el, k) => el.classList.toggle('is-current', k === f - 1));
    cur = i; frag = f;
    if (counter) counter.textContent = `${i + 1} / ${slides.length}`;
    if (bar) bar.style.transform = `scaleX(${(i + 1) / slides.length})`;
    history.replaceState(null, '', `#/${i + 1}${f ? `/${f}` : ''}`);
    if (broadcast) channel?.postMessage({ i, f });
    root.dispatchEvent(new CustomEvent('deck:change', { detail: { i, f } }));
  };
  const next = () => {
    const n = fragmentsOf(slides[cur]).length;
    if (frag < n) show(cur, frag + 1); else if (cur < slides.length - 1) show(cur + 1, 0);
  };
  const prev = () => {
    if (frag > 0) show(cur, frag - 1);
    else if (cur > 0) show(cur - 1, fragmentsOf(slides[cur - 1]).length);
  };

  // initial position from hash
  const m = location.hash.match(/^#\/(\d+)(?:\/(\d+))?/);
  show(m ? +m[1] - 1 : 0, m && m[2] ? +m[2] : 0, false);
  if (reduced) root.classList.add('is-reduced');

  channel?.addEventListener('message', (e) => { const { i, f } = e.data; show(i, f, false); });

  // overview
  const toggleOverview = (on = !root.classList.contains('is-overview')) => {
    root.classList.toggle('is-overview', on);
    if (on) {
      stage.style.transform = '';
      slides.forEach((s) => { s.inert = false; s.tabIndex = 0; });
      slides[cur].focus();
    } else {
      slides.forEach((s, k) => { s.inert = k !== cur; s.removeAttribute('tabindex'); });
      fit();
    }
  };
  slides.forEach((s, k) => s.addEventListener('click', () => {
    if (root.classList.contains('is-overview')) { toggleOverview(false); show(k, 0); }
  }));

  const fullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else root.requestFullscreen?.().catch(() => {});
  };
  const speaker = () => {
    const u = new URL(location.href);
    u.searchParams.set('speaker', '1');
    window.open(u.toString(), 'deck-speaker', 'width=1280,height=800');
  };
  const help = root.querySelector<HTMLDialogElement>('[data-help]');

  let typed = '';
  document.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (/^\d$/.test(k)) { typed += k; return; }
    if (k === 'Enter' && typed) { show(+typed - 1, 0); typed = ''; e.preventDefault(); return; }
    typed = '';
    if (root.classList.contains('is-overview') && k === 'Escape') { toggleOverview(false); return; }
    switch (k) {
      case 'ArrowRight': case 'ArrowDown': case ' ': case 'PageDown': case 'n': e.preventDefault(); next(); break;
      case 'ArrowLeft': case 'ArrowUp': case 'PageUp': case 'Backspace': e.preventDefault(); prev(); break;
      case 'Home': show(0); break;
      case 'End': show(slides.length - 1); break;
      case 'f': case 'F': fullscreen(); break;
      case 'o': case 'O': toggleOverview(); break;
      case 's': case 'S': if (!isSpeaker) speaker(); break;
      case 'p': case 'P': { const u = new URL(location.href); u.searchParams.set('print', '1'); u.hash = ''; window.open(u.toString()); break; }
      case '?': help?.open ? help.close() : help?.showModal(); break;
    }
  });

  root.querySelector('[data-next]')?.addEventListener('click', next);
  root.querySelector('[data-prev]')?.addEventListener('click', prev);
  root.querySelector('[data-full]')?.addEventListener('click', fullscreen);
  root.querySelector('[data-over]')?.addEventListener('click', () => toggleOverview());
  root.querySelector('[data-speaker]')?.addEventListener('click', speaker);
  root.querySelector('[data-helpbtn]')?.addEventListener('click', () => help?.showModal());

  // click on the right/left part of a slide advances/retreats (ignores interactive elements)
  stage.addEventListener('click', (e) => {
    if (root.classList.contains('is-overview')) return;
    const t = e.target as HTMLElement;
    if (t.closest('a, button, input, select, label, video, canvas, [data-interactive]')) return;
    const b = root.getBoundingClientRect();
    if (e.clientX - b.left > b.width * 0.35) next(); else prev();
  });

  // touch swipe
  let x0: number | null = null;
  root.addEventListener('touchstart', (e) => { x0 = e.touches[0].clientX; }, { passive: true });
  root.addEventListener('touchend', (e) => {
    if (x0 === null) return;
    const dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 50) (dx < 0 ? next : prev)();
    x0 = null;
  });

  if (isSpeaker) initSpeaker(root, slides, () => cur, () => frag, show);
}

function initSpeaker(root: HTMLElement, slides: HTMLElement[], getCur: () => number, getFrag: () => number, show: (i: number, f?: number) => void) {
  document.documentElement.classList.add('speaker-mode');
  const panel = document.createElement('div');
  panel.className = 'speaker';
  panel.innerHTML = `
    <div class="speaker__next"><p class="label">Next</p><div class="speaker__nextbox"></div></div>
    <div class="speaker__notes"><p class="label">Notes</p><div class="speaker__text"></div></div>
    <div class="speaker__meta"><span class="speaker__clock"></span><span class="speaker__timer">00:00</span><button class="btn" type="button">Reset timer</button></div>`;
  document.body.appendChild(panel);
  const nextBox = panel.querySelector<HTMLElement>('.speaker__nextbox')!;
  const text = panel.querySelector<HTMLElement>('.speaker__text')!;
  const update = () => {
    const i = getCur();
    const notes = slides[i].querySelector('.notes');
    text.innerHTML = notes ? notes.innerHTML : '<p class="muted">No notes for this slide.</p>';
    nextBox.replaceChildren();
    const n = slides[i + 1];
    if (n) {
      const c = n.cloneNode(true) as HTMLElement;
      c.classList.add('is-active', 'is-clone');
      c.querySelectorAll('[data-fragment]').forEach((f) => f.classList.add('is-visible'));
      c.querySelectorAll('video').forEach((v) => v.remove());
      const wrap = document.createElement('div');
      wrap.className = 'speaker__scale';
      wrap.appendChild(c);
      nextBox.appendChild(wrap);
      wrap.style.transform = `scale(${nextBox.clientWidth / 1600})`;
    } else nextBox.textContent = 'End of presentation';
  };
  root.addEventListener('deck:change', update);
  update();
  let t0 = Date.now();
  panel.querySelector('button')!.addEventListener('click', () => (t0 = Date.now()));
  const clock = panel.querySelector<HTMLElement>('.speaker__clock')!;
  const timer = panel.querySelector<HTMLElement>('.speaker__timer')!;
  setInterval(() => {
    const s = Math.floor((Date.now() - t0) / 1000);
    timer.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    clock.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }, 1000);
  void getFrag; void show;
}
