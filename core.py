"""Motion Deck: manifest -> scenes (grid-reveal zoom, split, single, title card) -> ffmpeg mp4.

No ComfyUI imports; tested offline. Frames are composed with PIL sub-pixel box resizes (no ffmpeg
zoompan, which jitters) and streamed to ffmpeg's stdin.
"""
import difflib
import math
import os
import re
import shutil
import subprocess
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from . import speaker
except ImportError:    # loaded outside a package (offline tests)
    import speaker

UA = {"User-Agent": "Mozilla/5.0 (MotionDeck)"}
IMG_HOLD = 2.5       # default seconds a still panel / scene is held
VID_HOLD_MAX = 6.0   # default cap for a video panel's hold
MOVE = 0.9           # seconds for a camera move between wide view and a panel
WIDE_HOLD = 0.7
FADE_DEFAULT = 0.5
TRANS = ("fade", "cut", "slide", "wipe", "slideup", "slidedown", "dip", "flash", "iris", "zoomthru", "blurfade")
CAMS = ("push", "pull", "panleft", "panright", "drift")
ANIMS = ["rise", "pop", "slide", "type", "wipe", "zoom", "drop", "blur"]

STYLES = [   # bg, gutter/border, title fill, title stroke, accent
    {"bg": (14, 16, 22), "title": (245, 245, 245), "stroke": (0, 0, 0), "accent": (240, 190, 40)},
    {"bg": (238, 236, 230), "title": (20, 20, 24), "stroke": (255, 255, 255), "accent": (200, 60, 40)},
    {"bg": (24, 10, 40), "title": (255, 240, 255), "stroke": (30, 0, 50), "accent": (255, 90, 200)},
    {"bg": (8, 30, 34), "title": (230, 255, 250), "stroke": (0, 20, 24), "accent": (60, 230, 200)},
]


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_duration(path):
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", path])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def resolve_file(name, input_dir, tmp_dir):
    name = (name or "").strip()
    if not name:
        return None
    if name.startswith(("http://", "https://")):
        ext = os.path.splitext(name.split("?")[0])[-1] or ".bin"
        path = os.path.join(tmp_dir, f"dl_{abs(hash(name)) % 10**8}{ext}")
        with urllib.request.urlopen(urllib.request.Request(name, headers=UA), timeout=120) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
        return path
    for cand in (os.path.join(input_dir, name) if input_dir else None, name):
        if cand and os.path.isfile(cand):
            return cand
    return None


# ── manifest ──────────────────────────────────────────────────────────────────

class Scene:
    def __init__(self):
        self.layout = None      # grid | split | single | card
        self.mode = None        # reveal | static
        self.media = []         # ("i"|"v", index 1-8)
        self.hold = None
        self.trans = "fade"
        self.trans_dur = None
        self.cam = "push"
        self.cols = None
        self.title = ""
        self.sub = ""
        self.start = 0.0          # speaker scenes: seconds into the source
        self.fixed = None         # speaker scenes: lock to person N (1 = leftmost)
        self.stack = False
        self.lang = ""
        self.whisper = None
        self.caps = {}
        self.anim = "rise"        # title entrance, see ANIMS
        self.cam_auto = True      # True until a camera word is given (voiceover slides then vary it)
        self.lead = 0.12          # voiceover: seconds a transition starts before its cue word
        self.cues = []            # voiceover scenes: list of cue Scenes
        self.cue_words, self.cue_at, self.until = [], None, []


def _num(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


def _apply_tokens(sc, head, notes):
    toks = head.replace(",", " ").split()
    low = [t.lower() for t in toks]
    i = 0
    while i < len(low):
        t = low[i]
        nxt = _num(low[i + 1]) if i + 1 < len(low) else None
        m = re.fullmatch(r"([iv])([1-8])", t)
        if m:
            sc.media.append((m.group(1), int(m.group(2))))
        elif t in ("grid", "split", "single", "card", "speaker", "voiceover"):
            sc.layout = t
        elif t in ("reveal", "static"):
            sc.mode = t
        elif t in ("hold", "for") and nxt is not None:
            sc.hold = max(0.5, min(300.0, nxt))
            i += 1
        elif t == "cols" and nxt is not None:
            sc.cols = max(1, min(4, int(nxt)))
            i += 1
        elif t in TRANS:
            sc.trans = t
            if nxt is not None:
                sc.trans_dur = max(0.1, min(3.0, nxt))
                i += 1
        elif t in CAMS:
            sc.cam = t
            sc.cam_auto = False
        elif t == "start" and nxt is not None:
            sc.start = max(0.0, nxt)
            i += 1
        elif t == "fixed" and nxt is not None:
            sc.fixed = max(1, int(nxt))
            i += 1
        elif t == "stack":
            sc.stack = True
        elif t == "anim" and i + 1 < len(low) and low[i + 1] in ANIMS + ["cycle"]:
            sc.anim = low[i + 1]
            i += 1
        elif t == "caps" and i + 1 < len(low) and low[i + 1] in ("karaoke", "single", "off"):
            sc.caps["caps"] = low[i + 1]
            i += 1
        elif t in ("font", "size", "words", "capspos", "stroke") and nxt is not None:
            sc.caps[t] = nxt
            i += 1
        elif t in ("hl", "fg") and i + 1 < len(low) and low[i + 1] in speaker.COLORS:
            sc.caps[t] = low[i + 1]
            i += 1
        elif t == "case" and i + 1 < len(low) and low[i + 1] in ("upper", "lower", "normal"):
            sc.caps["case"] = low[i + 1]
            i += 1
        elif t == "lead" and nxt is not None:
            sc.lead = max(0.0, min(1.0, nxt))
            i += 1
        elif t == "lang" and i + 1 < len(low):
            sc.lang = low[i + 1]
            i += 1
        elif t == "whisper" and i + 1 < len(low):
            sc.whisper = low[i + 1]
            i += 1
        else:
            notes.append(f"ignored word '{toks[i]}'")
        i += 1


_UNTIL_STOP = r"anim|hold|for|show|lead|" + "|".join(TRANS + CAMS) + r"|[iv][1-8]"


def _clamp_hold(v):
    return max(0.5, min(300.0, v))


def _parse_cue(chunk, notes):
    """cue <words | start | at N> show <i1|v1> [camera] [transition [s]] [anim X] [until <words>]
    [title: ...] [sub: ...]  ->  a Scene carrying cue_words / cue_at / until."""
    fields = list(re.finditer(r"(?i)\b(title|sub)\s*:", chunk))
    head = chunk[:fields[0].start()] if fields else chunk
    sc = Scene()
    sc.layout = "single"
    for k, m in enumerate(fields):
        end = fields[k + 1].start() if k + 1 < len(fields) else len(chunk)
        val = chunk[m.end():end].strip()
        if m.group(1).lower() == "title":
            sc.title = val
        else:
            sc.sub = val
    m = re.search(r"(?i)\bshow\b", head)
    trig, rest = (head[:m.start()], head[m.end():]) if m else (head, "")
    if not m:
        notes.append(f"cue '{trig.strip()[:30]}' has no 'show' (shows nothing)")
    tw = re.sub(r"[^\w\s']", " ", trig.lower()).split()
    if tw[:1] == ["start"]:
        sc.cue_at = 0.0
    elif tw[:1] == ["at"] and len(tw) > 1 and _num(tw[1]) is not None:
        sc.cue_at = max(0.0, _num(tw[1]))
    else:
        sc.cue_words = tw
    um = re.search(r"(?i)\buntil\s+(.+?)(?=\b(?:" + _UNTIL_STOP + r")\b|$)", rest)
    if um:
        sc.until = re.sub(r"[^\w\s']", " ", um.group(1).lower()).split()
        rest = rest[:um.start()] + " " + rest[um.end():]
    _apply_tokens(sc, rest, notes)
    if not sc.media:
        sc.layout = "card"
    return sc


def _parse_reel(head, title, sub, notes):
    """reel i1 3 push i2 2.5 drift slideup 0.6 v1 6 cut ...: each media token starts its own slide;
    a bare number is its seconds, then camera / transition [seconds] / anim words."""
    low = [t.lower() for t in head.replace(",", " ").split()][1:]
    slides, cur, i = [], None, 0
    while i < len(low):
        t = low[i]
        nxt = _num(low[i + 1]) if i + 1 < len(low) else None
        m = re.fullmatch(r"([iv])([1-8])", t)
        num = re.fullmatch(r"(\d+(?:\.\d+)?)s?", t)
        if m:
            cur = Scene()
            cur.layout = "single"
            cur.media = [(m.group(1), int(m.group(2)))]
            slides.append(cur)
        elif cur is None:
            notes.append(f"ignored word '{t}' before the first media in reel")
        elif num:
            cur.hold = _clamp_hold(float(num.group(1)))
        elif t in TRANS:
            cur.trans = t
            if nxt is not None:
                cur.trans_dur = max(0.1, min(3.0, nxt))
                i += 1
        elif t in CAMS:
            cur.cam = t
            cur.cam_auto = False
        elif t == "anim" and i + 1 < len(low) and low[i + 1] in ANIMS + ["cycle"]:
            cur.anim = low[i + 1]
            i += 1
        else:
            notes.append(f"ignored word '{t}' in reel")
        i += 1
    if slides:
        slides[0].title, slides[0].sub = title, sub
    return slides


def _norm_word(w):
    return re.sub(r"[^a-z0-9']", "", (w or "").lower())


def _same_word(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    k = min(len(a), len(b))
    if k >= 4 and a[:k - 1] == b[:k - 1]:     # lighthouse / lighthouses
        return True
    return k >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.84


def _find_phrase(words, phrase, pos):
    ph = [_norm_word(w) for w in phrase if _norm_word(w)]
    if not ph:
        return None
    ws = [_norm_word(w["word"]) for w in words]
    for j in range(max(0, pos), len(ws) - len(ph) + 1):
        if all(_same_word(ws[j + k], ph[k]) for k in range(len(ph))):
            return j
    return None


def expand_voiceover(sc, audio_src, input_dir, work, fps, notes):
    """Turn a voiceover scene (cues keyed to spoken words) into ordinary timed slides. Returns
    (slide Scenes, narration (path, start, dur) or None, report list)."""
    cues = sc.cues
    path = resolve_file(audio_src, input_dir, work) if audio_src else None
    if not cues:
        notes.append("voiceover scene has no cue")
        return [], None, []
    if not path:
        notes.append("voiceover needs an audio track (audio1 or reply to audio): cues spaced evenly instead")
    dur = probe_duration(path) if path else 0.0
    if sc.hold:
        dur = min(dur, sc.hold) if dur else sc.hold
    if not dur:
        dur = 3.0 * len(cues)
    words = []
    if path and any(c.cue_words or c.until for c in cues):
        try:
            samples = speaker.decode_wav16k(path, 0.0, dur, work)
            words = speaker.transcribe_words(samples, sc.lang, sc.whisper) if samples is not None else []
        except Exception as e:
            notes.append(f"voiceover transcription failed: {type(e).__name__}: {e}")
    entries, report, pos, prev = [], [], 0, 0.0
    for n, c in enumerate(cues):
        label = " ".join(c.cue_words) or (f"at {c.cue_at:g}" if c.cue_at is not None else "?")
        if c.cue_at is not None:
            t = c.cue_at
            pos = next((j for j, w in enumerate(words) if w["start"] >= t), pos)
            how = "time"
        elif c.cue_words and words:
            j = _find_phrase(words, c.cue_words, pos)
            if j is None:
                t = prev + max(1.0, (dur - prev) / max(1, len(cues) - n))
                notes.append(f"cue '{label}' not heard in the voice over: placed at {t:.1f}s")
                how = "not heard"
            else:
                t = float(words[j]["start"])
                pos = j + len(c.cue_words)
                how = "heard"
        else:
            t = dur * n / len(cues)
            how = "even"
        if entries:
            t = max(t, prev + 0.4)
        t = min(t, max(0.0, dur - 0.4))
        entries.append([t, c])
        report.append({"cue": label, "at": round(t, 2), "how": how, "show": " ".join(f"{k}{i}" for k, i in c.media)})
        prev = t
    entries[0][0] = 0.0     # the first visual is up from the start
    # 'until <word>': the slide ends at that word and a plain card fills the gap to the next cue
    full = []
    for n, (t, c) in enumerate(entries):
        full.append([t, c])
        if c.until and words:
            j = _find_phrase(words, c.until, 0)
            while j is not None and float(words[j]["start"]) <= t + 0.3:
                j = _find_phrase(words, c.until, j + 1)
            nxt_t = entries[n + 1][0] if n + 1 < len(entries) else dur
            if j is not None and float(words[j]["start"]) < nxt_t - 0.4:
                gap = Scene()
                gap.layout = "card"
                gap.trans, gap.trans_dur = "fade", 0.3
                full.append([float(words[j]["start"]), gap])
    starts = [0.0] + [max(0.0, t - c.lead) for t, c in full[1:]]
    total = dur + (0.8 if path else 0.0)
    F = [int(round(x * fps)) for x in starts] + [int(round(total * fps))]
    ov = [0]
    for i in range(1, len(full)):
        c = full[i][1]
        o = 0 if c.trans == "cut" else int(round((c.trans_dur or FADE_DEFAULT) * fps))
        ov.append(max(0, min(o, (F[i] - F[i - 1]) // 2)))
    slides = []
    for i, (t, c) in enumerate(full):
        n_i = F[i + 1] - F[i] + (ov[i + 1] if i + 1 < len(full) else 0)
        c.hold = max(2, n_i) / fps
        if c.cam_auto:
            c.cam = CAMS[i % len(CAMS)]
        if ov[i]:
            c.trans_dur = ov[i] / fps
        if i == 0:
            c.trans = "cut"
        slides.append(c)
    return slides, ((path, 0.0, total) if path else None), report


def parse_manifest(text, n_images, n_videos):
    """Plain-text grammar (survives Telegram: no slashes, brackets or pipes). Scenes start at the word
    'scene'; the options before 'title:' are space separated tokens; title: and sub: run to the next
    field or the end of the scene."""
    text = (text or "").strip()
    scenes, notes = [], []
    if text:
        parts = re.split(r"(?i)\bscene\b", text)
        chunks = [p for p in parts[1:]] if len(parts) > 1 else [text]
    else:
        chunks = []
    for ch in chunks:
        pieces = re.split(r"(?i)\bcue\b", ch)
        ch, cue_chunks = pieces[0], pieces[1:]
        fields = list(re.finditer(r"(?i)\b(title|sub)\s*:", ch))
        head = ch[:fields[0].start()] if fields else ch
        sc = Scene()
        for k, m in enumerate(fields):
            end = fields[k + 1].start() if k + 1 < len(fields) else len(ch)
            val = ch[m.end():end].strip()
            if m.group(1).lower() == "title":
                sc.title = val
            else:
                sc.sub = val
        if head.split()[:1] and head.split()[0].lower() == "reel":
            scenes.extend(_parse_reel(head, sc.title, sc.sub, notes))
            continue
        _apply_tokens(sc, head, notes)
        if cue_chunks:
            sc.layout = "voiceover"
            sc.cues = [_parse_cue(c, notes) for c in cue_chunks]
        scenes.append(sc)
    if not scenes:    # no manifest: groups of 4 as grid-reveal scenes
        media = [("i", i) for i in range(1, n_images + 1)] + [("v", i) for i in range(1, n_videos + 1)]
        for k in range(0, len(media), 4):
            sc = Scene()
            sc.media = media[k:k + 4]
            scenes.append(sc)
    for k, sc in enumerate(scenes):
        if sc.anim == "cycle":
            sc.anim = ANIMS[k % len(ANIMS)]
        if sc.layout is None:
            sc.layout = "card" if not sc.media else ("single" if len(sc.media) == 1 else "grid")
        if sc.layout == "single":
            sc.media = sc.media[:1]
        if sc.layout == "split":
            sc.media = sc.media[:2]
        if sc.layout == "speaker":
            sc.media = sc.media[:1]
        if sc.mode is None:
            sc.mode = "reveal" if (sc.layout == "grid" and len(sc.media) > 1) else "static"
        sc.media = sc.media[:9]
    return scenes, notes


# ── fonts / text ──────────────────────────────────────────────────────────────

def _font(size, bold=True):
    p = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


_PFONTS = {"xb": "Poppins-ExtraBold.ttf", "sb": "Poppins-SemiBold.ttf", "md": "Poppins-Medium.ttf"}


def _pfont(kind, size):
    p = os.path.join(os.path.dirname(__file__), "fonts", _PFONTS[kind])
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return _font(size, bold=(kind != "md"))


def _back(x):
    x = min(1.0, max(0.0, x))
    return 1.0 + 2.70158 * (x - 1.0) ** 3 + 1.70158 * (x - 1.0) ** 2


def _bounce(x):
    n, d = 7.5625, 2.75
    if x < 1 / d:
        return n * x * x
    if x < 2 / d:
        x -= 1.5 / d
        return n * x * x + 0.75
    if x < 2.5 / d:
        x -= 2.25 / d
        return n * x * x + 0.9375
    x -= 2.625 / d
    return n * x * x + 0.984375


def _out(x):
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


class TitleAnim:
    """Title + subtitle that animate line by line: each line slides up from behind its own baseline
    (clipped to its line box), staggered, then the whole block fades and lifts away. Cards get an
    accent rule that wipes open under the text; titles over media sit on a dark pill with an accent bar."""

    def __init__(self, title, sub, W, H, style, card=False, top=False, anim="rise"):
        self.W, self.H, self.card = W, H, card
        self.anim = anim if anim in ANIMS else "rise"
        self.items = []     # dict(img, x, y, kind, delay)
        d = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        base = H * (0.075 if card else 0.048 if top else 0.066)
        tsize = max(22, int(base if not card else H * 0.085))
        longest = max((d.textlength(w, font=_pfont("xb", tsize)) for w in (title or "x").split()), default=0)
        maxw = int(W * (0.84 if card else 0.78))
        if longest > maxw:
            tsize = max(20, int(tsize * maxw / longest))
        ssize = max(16, int(tsize * 0.5))
        tf, sf = _pfont("xb", tsize), _pfont("md", ssize)
        tl = _wrap(d, title, tf, maxw) if title else []
        sl = _wrap(d, sub, sf, maxw) if sub else []
        fill = style["title"] if card else (250, 250, 250)
        sfill = style["accent"]
        tlh = int(tsize * 1.28)
        slh = int(ssize * 1.5)
        gap = int(ssize * 0.55) if (tl and sl) else 0
        rule_h = max(4, tsize // 11) if (card and tl) else 0
        rule_gap = int(tsize * 0.35) if rule_h else 0
        block_h = tlh * len(tl) + (gap + slh * len(sl) if sl else 0) + (rule_gap + rule_h if rule_h else 0)
        widths = [d.textlength(ln, font=tf) for ln in tl] + [d.textlength(ln, font=sf) for ln in sl]
        block_w = int(max(widths)) if widths else 0
        if card:
            y = (H - block_h) // 2
        else:
            pad = int(tsize * 0.55)
            bottom = int(H * (0.93 if top else 0.88))
            y = bottom - block_h - pad
            px0, py0 = (W - block_w) // 2 - pad, y - pad
            pill = Image.new("RGBA", (block_w + 2 * pad, block_h + 2 * pad), (0, 0, 0, 0))
            pd = ImageDraw.Draw(pill)
            pd.rounded_rectangle((0, 0, pill.width - 1, pill.height - 1), radius=int(pad * 0.9), fill=(10, 10, 14, 175))
            bar = max(4, tsize // 10)
            pd.rounded_rectangle((pad // 2, pad // 2, pad // 2 + bar, pill.height - pad // 2 - 1), radius=bar // 2, fill=sfill + (255,))
            self.items.append(dict(img=pill, x=px0, y=py0, kind="fade", delay=0.0))
        k = 0
        for ln in tl:
            self.items.append(self._line(d, ln, tf, tlh, fill, W, y, k, card, tsize))
            y += tlh
            k += 1
        y += gap
        for ln in sl:
            self.items.append(self._line(d, ln, sf, slh, sfill, W, y, k + 1, card, ssize))
            y += slh
            k += 1
        if rule_h:
            y += rule_gap
            rw = int(min(W * 0.5, max(block_w * 0.45, W * 0.12)))
            self.items.append(dict(img=Image.new("RGBA", (rw, rule_h), sfill + (255,)), x=(W - rw) // 2, y=y,
                                   kind="rule", delay=0.08 * (k + 1)))

        if self.anim == "type":    # lines are typed one after another
            t_end = 0.0
            for it in self.items:
                if it["kind"] == "line":
                    it["delay"] = t_end
                    t_end += it["dur"] + 0.08
                elif it["kind"] == "rule":
                    it["delay"] = t_end

    @staticmethod
    def _line(d, text, font, lh, fill, W, y, k, card, size):
        pad = max(4, size // 8)
        w = int(d.textlength(text, font=font)) + 2 * pad
        img = Image.new("RGBA", (w, lh), (0, 0, 0, 0))
        asc, desc = font.getmetrics()
        ImageDraw.Draw(img).text((pad, (lh - asc - desc) // 2), text, font=font, fill=fill + (255,),
                                 stroke_width=0 if card else 0)
        return dict(img=img, x=(W - w) // 2, y=y, kind="line", delay=0.085 * k, text=text, font=font, pad=pad,
                    dur=max(0.35, (0.03 if size > 40 else 0.02) * len(text)))

    def _line_frame(self, it, p, local):
        """(RGBA image, dx, dy) for one text line at eased progress p (0..1) under this entrance."""
        src = it["img"]
        w, h = src.size
        k = self.anim
        if k == "rise":
            tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            tmp.paste(src, (0, int((1.0 - p) * h * 1.05)))
            return tmp, 0, 0
        if k == "drop":
            q = _bounce(min(1.0, max(0.0, (local - it["delay"]) / 0.75)))
            return src, 0, int((q - 1.0) * h * 1.6)
        if k == "slide":
            sign = -1 if (int(round(it["delay"] / 0.085)) % 2) else 1
            return src, int(sign * (1.0 - p) * self.W * 0.35), 0
        if k in ("pop", "zoom"):
            s0 = 0.55 if k == "pop" else 1.7
            sc = (_back(p) if k == "pop" else p)
            sc = s0 + (1.0 - s0) * sc
            nw, nh = max(2, int(w * sc)), max(2, int(h * sc))
            return src.resize((nw, nh), Image.BILINEAR), (w - nw) // 2, (h - nh) // 2
        if k == "blur":
            r = (1.0 - p) * 14.0
            return (src.filter(ImageFilter.GaussianBlur(r)) if r > 0.4 else src), 0, 0
        if k == "wipe":
            import numpy as np
            a = np.array(src.getchannel("A"), np.float32)
            edge = p * (w + 60) - 30
            ramp = np.clip((edge - np.arange(w, dtype=np.float32)) / 30.0, 0.0, 1.0)
            out = src.copy()
            out.putalpha(Image.fromarray((a * ramp[None, :]).astype("uint8"), "L"))
            return out, 0, 0
        if k == "type":
            q = min(1.0, max(0.0, (local - it["delay"]) / it["dur"]))
            n = int(round(q * len(it["text"])))
            vis = it["pad"] + int(ImageDraw.Draw(Image.new("RGB", (4, 4))).textlength(it["text"][:n], font=it["font"]))
            tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            tmp.paste(src.crop((0, 0, min(w, vis), h)), (0, 0))
            if q < 1.0 and int(local * 6) % 2 == 0:    # blinking cursor
                cw = max(3, h // 18)
                ImageDraw.Draw(tmp).rectangle((min(w - cw, vis + 2), int(h * 0.18), min(w, vis + 2 + cw), int(h * 0.82)),
                                              fill=(255, 255, 255, 255))
            return tmp, 0, 0
        return src, 0, 0

    def draw(self, img, t, t0, end, out=0.35):
        local = t - t0
        q = min(1.0, max(0.0, (t - (end - out)) / out))
        for it in self.items:
            dur_in = it.get("dur", 0.55) if (self.anim == "type" and it["kind"] == "line") else 0.55
            p = _out((local - it["delay"]) / (0.55 if self.anim != "type" else dur_in))
            if p <= 0.0 or q >= 1.0:
                continue
            src = it["img"]
            a = (1.0 if (self.anim == "type" and it["kind"] == "line") else min(1.0, p * 1.8)) * (1.0 - q)
            lift = -int(q * self.H * 0.012)
            dx = dy = 0
            if it["kind"] == "line":
                tmp, dx, dy = self._line_frame(it, p if self.anim != "type" else 1.0, local)
            elif it["kind"] == "rule":
                w, h = src.size
                tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                ww = max(1, int(w * p))
                tmp.paste(src.crop((0, 0, ww, h)), ((w - ww) // 2, 0))
            else:
                tmp = src.copy()
            if a < 1.0:
                if tmp is src:
                    tmp = tmp.copy()    # never dim the cached line image in place
                tmp.putalpha(tmp.getchannel("A").point(lambda v, a=a: int(v * a)))
            img.paste(tmp, (it["x"] + dx, it["y"] + lift + dy), tmp)


def text_layer(title, sub, W, H, style, card=False, top=False, anim="rise"):
    return TitleAnim(title, sub, W, H, style, card=card, top=top, anim=anim)


def _gradient(W, H, bg):
    import numpy as np
    lum = sum(bg) / 3.0
    d = 14 if lum < 128 else -10
    top = np.array(bg, np.float32)
    bot = np.clip(top + d, 0, 255)
    col = (top[None, :] * (1 - np.linspace(0, 1, H)[:, None]) + bot[None, :] * np.linspace(0, 1, H)[:, None])
    return Image.fromarray(np.repeat(col[:, None, :], W, axis=1).astype("uint8"), "RGB")


# ── media ─────────────────────────────────────────────────────────────────────

def cover(img, w, h):
    iw, ih = img.size
    s = max(w / iw, h / ih)
    img = img.resize((max(w, round(iw * s)), max(h, round(ih * s))), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


class VideoReader:
    """Sequential rgb24 reader at a fixed size/fps. Holds the first frame until read starts and
    the last frame once the clip ends (freeze)."""

    def __init__(self, path, w, h, fps, max_seconds):
        self.w, self.h, self.fps = w, h, fps
        self.path = path
        self.max_n = max(1, int(max_seconds * fps))
        self.n = 0
        self.eof = False
        self.proc = None
        self.last = None
        self.first = self._grab_first()

    def _grab_first(self):
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", self.path, "-frames:v", "1", "-vf", self._vf(),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        buf = r.stdout
        if len(buf) < self.w * self.h * 3:
            raise RuntimeError(f"cannot decode {os.path.basename(self.path)}: {r.stderr[-200:]}")
        return Image.frombytes("RGB", (self.w, self.h), buf[:self.w * self.h * 3])

    def _vf(self):
        return (f"fps={self.fps},scale={self.w}:{self.h}:force_original_aspect_ratio=increase,"
                f"crop={self.w}:{self.h}")

    def next(self):
        if self.eof:
            return self.last or self.first
        if self.proc is None:
            self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-i", self.path, "-vf", self._vf(), "-frames:v", str(self.max_n),
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if self.proc is not None:
            buf = self.proc.stdout.read(self.w * self.h * 3)
            if len(buf) == self.w * self.h * 3:
                self.last = Image.frombytes("RGB", (self.w, self.h), buf)
                return self.last
            self.close()
            self.eof = True
        return self.last or self.first

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdout.close()
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
        self.proc = None


class Panel:
    def __init__(self, kind, path, pw, ph, fps, hold):
        self.kind, self.pw, self.ph = kind, pw, ph
        self.start = self.end = 0.0   # seconds within the scene during which the clip plays
        self.reader = None
        if kind == "i":
            self.still = cover(Image.open(path).convert("RGB"), pw, ph)
        else:
            self.reader = VideoReader(path, pw, ph, fps, hold)
            self.still = self.reader.first
        self.finished = False

    def frame_at(self, t):
        if self.reader is None:
            return self.still
        if t < self.start:
            return self.still
        if self.finished or t >= self.end:
            self.finished = True
            self.reader.close()
            return self.reader.last or self.still
        return self.reader.next()


# ── camera ────────────────────────────────────────────────────────────────────

def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def fit_box(x0, y0, x1, y1, aspect):
    """Grow (x0,y0,x1,y1) about its centre to the output aspect ratio."""
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def lerp_box(a, b, e):
    """Interpolate centre linearly and size geometrically (feels like a real zoom, not a crop swell)."""
    acx, acy, aw = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2, a[2] - a[0]
    bcx, bcy, bw = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0]
    ah, bh = a[3] - a[1], b[3] - b[1]
    w = aw * (bw / aw) ** e
    h = ah * (bh / ah) ** e
    cx, cy = acx + (bcx - acx) * e, acy + (bcy - acy) * e
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class SceneRender:
    """One scene: build() lays out the canvas and camera timeline, frame(k) returns an RGB PIL image."""

    def __init__(self, sc, paths, W, H, fps, style, pace):
        self.sc, self.W, self.H, self.fps, self.style = sc, W, H, fps, style
        self.aspect = W / H
        self.panels = []
        for kind, idx in sc.media:
            p = paths.get((kind, idx))
            if not p:
                continue
            self.panels.append((kind, p))
        n = len(self.panels)
        self.move, self.wide_hold = MOVE * pace, WIDE_HOLD * pace
        if sc.layout == "card" or n == 0:
            self.kind = "card"
            self.duration = (sc.hold or 3.0) * 1.0
            self.layer = text_layer(sc.title, sc.sub, W, H, style, card=True, anim=sc.anim)
            self.bgimg = _gradient(W, H, style["bg"])
            self.panel_objs = []
            self.timeline = []
            self.canvas = None
            return
        cols = sc.cols or (1 if sc.layout == "single" else 2 if sc.layout == "split" else
                           (1 if n == 1 else 2 if n <= 4 else 3))
        if sc.layout == "split":
            cols = n
        rows = math.ceil(n / cols)
        gap = int(W * 0.03) if (sc.layout in ("grid", "split") and n > 1) else 0
        pw, ph = W, H
        if sc.layout == "grid" and n > 1:
            pw, ph = W, H
        self.pw, self.ph, self.gap, self.cols, self.rows = pw, ph, gap, cols, rows
        # play windows and camera keyframes
        self.panel_objs = []
        for kind, p in self.panels:
            hold = sc.hold or (IMG_HOLD if kind == "i" else min(VID_HOLD_MAX, max(1.0, probe_duration(p))))
            self.panel_objs.append(Panel(kind, p, pw, ph, fps, hold))
            self.panel_objs[-1].hold = hold
        cw = cols * pw + (cols + 1) * gap
        ch = rows * ph + (rows + 1) * gap
        self.layout_w, self.layout_h = cw, ch
        self.rects = []
        for k in range(n):
            r, c = divmod(k, cols)
            x = gap + c * (pw + gap)
            y = gap + r * (ph + gap)
            self.rects.append((x, y, x + pw, y + ph))
        wide = fit_box(0, 0, cw, ch, self.aspect)
        # canvas big enough to contain the wide box
        self.off = (max(0, -int(math.floor(wide[0]))), max(0, -int(math.floor(wide[1]))))
        cwid = int(math.ceil(max(cw, wide[2]))) + self.off[0]
        chei = int(math.ceil(max(ch, wide[3]))) + self.off[1]
        self.canvas = Image.new("RGB", (cwid, chei), style["bg"])
        self.rects = [(a + self.off[0], b + self.off[1], c_ + self.off[0], d_ + self.off[1]) for a, b, c_, d_ in self.rects]
        wide = tuple(v + o for v, o in zip(wide, (self.off[0], self.off[1], self.off[0], self.off[1])))
        self.wide = wide
        for rect, po in zip(self.rects, self.panel_objs):
            self.canvas.paste(po.still, (rect[0], rect[1]))
        # timeline: list of (t_start, t_end, box_from, box_to)
        t = 0.0
        tl = []
        if sc.layout == "grid" and sc.mode == "reveal" and n > 1:
            tl.append((t, t + self.wide_hold, wide, wide))
            t += self.wide_hold
            for rect, po in zip(self.rects, self.panel_objs):
                tl.append((t, t + self.move, wide, rect))
                t += self.move
                po.start, po.end = t, t + po.hold
                tl.append((t, t + po.hold, rect, rect))
                t += po.hold
                tl.append((t, t + self.move, rect, wide))
                t += self.move
                tl.append((t, t + self.wide_hold, wide, wide))
                t += self.wide_hold
            self.kind = "reveal"
        elif sc.layout in ("grid", "split") and n > 1:
            dur = sc.hold or max(po.hold for po in self.panel_objs)
            for po in self.panel_objs:
                po.start, po.end = 0.0, dur
            tl.append((0.0, dur, wide, wide))
            t = dur
            self.kind = "static"
        else:   # single: camera drift over one panel
            po = self.panel_objs[0]
            dur = po.hold
            po.start, po.end = 0.0, dur
            r = self.rects[0]
            w, h = r[2] - r[0], r[3] - r[1]
            z = 1.12
            cw_, ch_ = w / z, h / z
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            full = r
            small = (cx - cw_ / 2, cy - ch_ / 2, cx + cw_ / 2, cy + ch_ / 2)
            cam = sc.cam
            if cam == "pull":
                a, b = small, full
            elif cam == "panleft":
                a = (r[0] + w - cw_, small[1], r[0] + w, small[3])
                b = (r[0], small[1], r[0] + cw_, small[3])
            elif cam == "panright":
                a = (r[0], small[1], r[0] + cw_, small[3])
                b = (r[0] + w - cw_, small[1], r[0] + w, small[3])
            elif cam == "drift":
                z2 = 1.04
                a = full
                b = (cx - w / z2 / 2, cy - h / z2 / 2, cx + w / z2 / 2, cy + h / z2 / 2)
            else:
                a, b = full, small
            tl.append((0.0, dur, a, b))
            t = dur
            self.kind = "single"
        self.timeline = tl
        self.duration = t
        self.layer = text_layer(sc.title, sc.sub, W, H, style, anim=sc.anim) if (sc.title or sc.sub) else None
        self.title_dur = min(self.duration - 0.3, max(2.0, 3.0)) if self.layer else 0.0

    def n_frames(self):
        return max(2, int(round(self.duration * self.fps)))

    def _box(self, t):
        for (a, b, f, to) in self.timeline:
            if a <= t < b:
                return lerp_box(f, to, _ease((t - a) / max(b - a, 1e-6))) if f != to else f
        return self.timeline[-1][3]

    def frame(self, k):
        t = k / self.fps
        if self.kind == "card":
            img = self.bgimg.copy()
            self._title(img, t, self.duration)
            return img
        for po, rect in zip(self.panel_objs, self.rects):
            if po.reader is not None:
                self.canvas.paste(po.frame_at(t), (rect[0], rect[1]))
        box = self._box(t)
        cwid, chei = self.canvas.size
        box = (max(0.0, box[0]), max(0.0, box[1]), min(float(cwid), box[2]), min(float(chei), box[3]))
        img = self.canvas.resize((self.W, self.H), Image.BILINEAR, box=box)
        if self.layer is not None:
            self._title(img, t, self.title_dur + 0.3)
        return img

    def _title(self, img, t, dur):
        if self.layer is None:
            return
        t0 = 0.3
        end = min(dur, t0 + (self.title_dur if self.kind != "card" else self.duration - 0.6))
        if t < t0 or t > end:
            return
        self.layer.draw(img, t, t0, end)

    def close(self):
        for po in self.panel_objs:
            if po.reader is not None:
                po.reader.close()


class SpeakerScene:
    """Landscape clip with 1-3 people -> vertical (or any aspect) crop that cuts to whoever is talking,
    with word-by-word captions. Analysis (whisper words, faces, speaker choice) runs in __init__."""
    kind = "speaker"
    _title = SceneRender._title

    def __init__(self, sc, path, W, H, fps, style, work):
        self.sc, self.W, self.H, self.fps, self.style = sc, W, H, fps, style
        self.panels = [path]
        self.notes = []
        t0 = time.time()
        total = probe_duration(path) or 1.0
        start = min(sc.start, max(0.0, total - 1.0))
        self.start = start
        self.duration = max(1.0, min(sc.hold or 90.0, total - start))
        self.audio = (path, start, self.duration) if speaker.has_audio(path) else None
        words = []
        samples = speaker.decode_wav16k(path, start, self.duration, work) if self.audio else None
        if samples is not None and len(samples) > 8000:
            try:
                words = speaker.transcribe_words(samples, sc.lang, sc.whisper)
            except Exception as e:     # keep the video even when ASR is unavailable
                self.notes.append(f"transcription failed: {type(e).__name__}: {e}"[:200])
        else:
            self.notes.append("no speech audio found")
        t_asr = time.time()
        try:
            times, frames, size = speaker.sample_faces(path, start, self.duration)
            self.tracks = speaker.build_tracks(times, frames, size)
        except Exception as e:
            self.tracks = {"K": 0, "x": np.zeros((0, 0)), "act": np.zeros((0, 0)), "times": np.zeros(0)}
            self.notes.append(f"face tracking failed: {type(e).__name__}: {e}"[:200])
        K = self.tracks["K"]
        if K == 0:
            self.notes.append("no faces found: centred crop")
        fixed = None if sc.fixed is None else sc.fixed - 1
        self.segs = speaker.decide_speakers(words, self.tracks, self.duration, fixed=fixed)
        self.stack = bool(sc.stack and K >= 2)
        self.words = words
        opts = dict(sc.caps)
        self.caps = None if opts.get("caps") == "off" else speaker.Captions(words, W, H, opts)
        # source decode size (long side <= 1920)
        sw, sh = speaker.probe_video(path)
        k = min(1.0, 1920.0 / max(sw, sh))
        self.sw, self.sh = max(2, int(sw * k) // 2 * 2), max(2, int(sh * k) // 2 * 2)
        self.path = path
        self.reader = None
        self.last = None
        hh = H / 2.0 if self.stack else float(H)
        asp = W / hh
        if self.sw / self.sh >= asp:
            self.ch, self.cw = float(self.sh), self.sh * asp
        else:
            self.cw, self.ch = float(self.sw), self.sw / asp
        self.layer = text_layer(sc.title, sc.sub, W, H, style, top=True, anim=sc.anim) if (sc.title or sc.sub) else None
        self.title_dur = min(3.0, self.duration - 0.3) if self.layer else 0.0
        self.info = {"words": len(words), "people": K, "stack": self.stack,
                     "speaker_segments": [(round(a, 2), round(b, 2), s + 1) for a, b, s in self.segs][:40],
                     "asr_seconds": round(t_asr - t0, 1), "analysis_seconds": round(time.time() - t0, 1)}

    def n_frames(self):
        return max(2, int(round(self.duration * self.fps)))

    def _read(self):
        if self.reader is None:
            self.reader = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-ss", f"{self.start:.3f}", "-t", f"{self.duration + 0.5:.3f}", "-i", self.path,
                 "-vf", f"fps={self.fps},scale={self.sw}:{self.sh}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        n = self.sw * self.sh * 3
        buf = self.reader.stdout.read(n)
        if len(buf) == n:
            self.last = Image.frombytes("RGB", (self.sw, self.sh), buf)
        return self.last or Image.new("RGB", (self.sw, self.sh), (0, 0, 0))

    def _cx(self, ident, t):
        tr = self.tracks
        if tr["K"] == 0 or ident >= tr["K"]:
            return 0.5 * self.sw
        return float(np.interp(t, tr["times"], tr["x"][ident])) * self.sw

    def _box(self, cx):
        x0 = min(max(cx - self.cw / 2, 0.0), self.sw - self.cw)
        y0 = (self.sh - self.ch) / 2
        return (x0, y0, x0 + self.cw, y0 + self.ch)

    def frame(self, k):
        t = k / self.fps
        src = self._read()
        ident = self.segs[-1][2]
        for a, b, s in self.segs:
            if a <= t < b:
                ident = s
                break
        if self.stack:
            half = self.H // 2
            img = Image.new("RGB", (self.W, self.H))
            order = sorted(range(self.tracks["K"]), key=lambda i: self._cx(i, t))[:2]
            for row, who in enumerate(order):
                part = src.resize((self.W, half), Image.BILINEAR, box=self._box(self._cx(who, t)))
                img.paste(part, (0, row * half))
                if who == ident:
                    img.paste(Image.new("RGB", (10, half), self.style["accent"]), (0, row * half))
        else:
            img = src.resize((self.W, self.H), Image.BILINEAR, box=self._box(self._cx(ident, t)))
        if self.caps is not None:
            self.caps.draw(img, t)
        if self.layer is not None:
            self._title(img, t, self.title_dur + 0.3)
        return img

    def close(self):
        if self.reader is not None:
            try:
                self.reader.stdout.close()
                self.reader.kill()
            except Exception:
                pass
            self.reader = None


# ── transitions ───────────────────────────────────────────────────────────────

def _zoom(img, s):
    W, H = img.size
    cw, ch = W / s, H / s
    return img.resize((W, H), Image.BILINEAR, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))


def blend(a, b, p, kind):
    p = _ease(p)
    W, H = a.size
    if kind == "slide":
        off = int(W * p)
        out = Image.new("RGB", (W, H))
        out.paste(a, (-off, 0))
        out.paste(b, (W - off, 0))
        return out
    if kind in ("slideup", "slidedown"):
        off = int(H * p)
        sign = 1 if kind == "slideup" else -1
        out = Image.new("RGB", (W, H))
        out.paste(a, (0, -sign * off))
        out.paste(b, (0, sign * (H - off)))
        return out
    if kind == "wipe":
        x = int(W * p)
        out = a.copy()
        out.paste(b.crop((0, 0, x, H)), (0, 0))
        return out
    if kind in ("dip", "flash"):
        col = (0, 0, 0) if kind == "dip" else (255, 255, 255)
        solid = Image.new("RGB", (W, H), col)
        return Image.blend(a, solid, p * 2) if p < 0.5 else Image.blend(solid, b, (p - 0.5) * 2)
    if kind == "iris":
        r = p * math.hypot(W, H) / 2
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse((W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r), fill=255)
        out = a.copy()
        out.paste(b, (0, 0), mask)
        return out
    if kind == "zoomthru":
        return Image.blend(_zoom(a, 1.0 + 0.35 * p), _zoom(b, 1.35 - 0.35 * p), p)
    if kind == "blurfade":
        return Image.blend(a.filter(ImageFilter.GaussianBlur(p * 12)), b.filter(ImageFilter.GaussianBlur((1 - p) * 12)), p)
    return Image.blend(a, b, p)


def mux_audio(silent, out_path, parts, bed, secs, work):
    """Lay each speaker scene's source audio at its start in the timeline, add the optional bed
    (ducked under speech), fade the end, and copy the picture stream untouched."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", silent]
    fl, labels, n = [], [], 1
    for (src, start, dur), off in parts:
        cmd += ["-i", src]
        ms = int(off * 1000)
        fl.append(f"[{n}:a]atrim=start={start:.3f}:duration={dur:.3f},asetpts=PTS-STARTPTS,aresample=44100,"
                  f"aformat=channel_layouts=stereo,adelay={ms}|{ms}[p{n}]")
        labels.append(f"[p{n}]")
        n += 1
    if bed:
        cmd += ["-i", bed]
        fl.append(f"[{n}:a]aresample=44100,aformat=channel_layouts=stereo,volume={0.25 if parts else 1.0}[p{n}]")
        labels.append(f"[p{n}]")
    m = len(labels)
    fade = 1.5 if (bed and not parts) else 0.25
    tail = f"[am]afade=t=out:st={max(0.0, secs - fade):.2f}:d={fade},atrim=duration={secs:.3f}[aout]"
    last = None
    for norm in (":normalize=0", ""):      # older ffmpeg builds lack amix normalize
        if m > 1:
            mix = f"{''.join(labels)}amix=inputs={m}{norm}:duration=longest" + ("" if norm else f",volume={m}") + "[am]"
        else:
            mix = f"{labels[0]}anull[am]"
        full = cmd + ["-filter_complex", ";".join(fl + [mix, tail]), "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy",
                      "-c:a", "aac", "-b:a", "160k", "-t", f"{secs:.3f}", "-movflags", "+faststart", out_path]
        last = subprocess.run(full, capture_output=True, text=True)
        if last.returncode == 0 or m == 1:
            break
    if last.returncode:
        raise RuntimeError("audio mux failed: " + last.stderr[-500:])


# ── compile ───────────────────────────────────────────────────────────────────

def compile_deck(manifest, vid_files, img_files, audio_src, out_path, work, input_dir,
                 width=0, height=0, fps=24, max_frames=2400, style_id=0, pace=1.0):
    W = int(width) or 0
    H = int(height) or 0
    first_speaker = bool(re.search(r"(?i)\bscene\s+speaker\b", manifest or ""))
    if not W and not H:
        W, H = (720, 1280) if first_speaker else (1280, 720)
    elif not W:
        W = int(round(H * 16 / 9))
    elif not H:
        H = int(round(W * 9 / 16))
    W, H = max(256, (W // 16) * 16), max(256, (H // 16) * 16)
    style = STYLES[int(style_id) % len(STYLES)]
    pace = min(2.0, max(0.5, float(pace) or 1.0))
    paths, notes = {}, []
    for i, v in enumerate(vid_files, 1):
        p = resolve_file(v, input_dir, work)
        if p:
            paths[("v", i)] = p
    for i, v in enumerate(img_files, 1):
        p = resolve_file(v, input_dir, work)
        if p:
            paths[("i", i)] = p
    n_img = sum(1 for k in paths if k[0] == "i")
    n_vid = sum(1 for k in paths if k[0] == "v")
    scenes, pnotes = parse_manifest(manifest, 0, 0)
    if not (manifest or "").strip():
        scenes, pnotes = parse_manifest("", max([k[1] for k in paths if k[0] == "i"] or [0]),
                                        max([k[1] for k in paths if k[0] == "v"] or [0]))
    notes += pnotes
    cue_report, narr_at, narr = [], None, None
    expanded = []
    for sc in scenes:
        if sc.layout == "voiceover":
            subs, nar, rep = expand_voiceover(sc, audio_src, input_dir, work, fps, notes)
            if nar and narr is None:
                narr_at, narr = len(expanded), nar
            cue_report += rep
            expanded += subs
        else:
            expanded.append(sc)
    scenes = expanded
    if narr:
        audio_src = None        # the voice over is laid in per scene below, not mixed as a music bed
    for sc in scenes:
        miss = [f"{k}{i}" for k, i in sc.media if (k, i) not in paths]
        if miss:
            notes.append(f"missing media {', '.join(miss)} (skipped)")
    renders = []
    try:
        for sc in scenes:
            vp = next((paths[m] for m in sc.media if m[0] == "v" and m in paths), None)
            if sc.layout == "speaker" and vp:
                renders.append(SpeakerScene(sc, vp, W, H, fps, style, work))
            else:
                if sc.layout == "speaker":
                    sc.layout = "card"
                    notes.append("speaker scene needs a video (v1..v8)")
                renders.append(SceneRender(sc, paths, W, H, fps, style, pace))
        if narr is not None and narr_at is not None and narr_at < len(renders):
            renders[narr_at].audio = narr
        # timeline offsets: scene i starts when the previous ends minus the transition overlap
        offs, ovs, g = [], [], 0
        for i, r in enumerate(renders):
            ov = 0
            if i > 0 and r.sc.trans != "cut":
                ov = int(round((r.sc.trans_dur or FADE_DEFAULT) * fps))
                ov = min(ov, renders[i - 1].n_frames() // 2, r.n_frames() // 2)
            g = g - ov if i > 0 else 0
            offs.append(g)
            ovs.append(ov)
            g += r.n_frames()
        total = min(g, int(max_frames))
        if total < g:
            notes.append(f"trimmed to {total} frames (length limit)")
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-"]
        audio = resolve_file(audio_src, input_dir, work) if audio_src else None
        silent = os.path.join(work, "silent.mp4")
        cmd += ["-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", silent]
        logf = open(os.path.join(work, "ffmpeg_out.log"), "wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=logf, stderr=logf)
        try:
            for gi in range(total):
                active = [(i, r) for i, r in enumerate(renders) if offs[i] <= gi < offs[i] + r.n_frames()]
                if not active:
                    active = [(len(renders) - 1, renders[-1])]
                i, r = active[0]
                img = r.frame(min(gi - offs[i], r.n_frames() - 1))
                if len(active) > 1:
                    j, r2 = active[1]
                    img2 = r2.frame(gi - offs[j])
                    p = (gi - offs[j]) / max(1, ovs[j])
                    img = blend(img, img2, min(1.0, p), r2.sc.trans)
                proc.stdin.write(img.tobytes())
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
        rc = proc.wait()
        logf.close()
        if rc:
            raise RuntimeError("ffmpeg failed: " + open(os.path.join(work, "ffmpeg_out.log")).read()[-600:])
        parts = [(r.audio, offs[i] / fps) for i, r in enumerate(renders) if getattr(r, "audio", None)]
        if parts or audio:
            mux_audio(silent, out_path, parts, audio, total / fps, work)
        else:
            shutil.move(silent, out_path)
    finally:
        for r in renders:
            r.close()
    return {"size": f"{W}x{H}", "fps": fps, "frames": total, "seconds": round(total / fps, 2),
            "scenes": [dict({"layout": r.sc.layout, "mode": r.sc.mode, "panels": len(r.panels),
                             "seconds": round(r.duration, 2), "title": r.sc.title[:60]},
                            **(getattr(r, "info", None) or {}), **({"notes": r.notes} if getattr(r, "notes", None) else {}))
                       for r in renders],
            **({"cues": cue_report} if cue_report else {}),
            "images": n_img, "videos": n_vid, "notes": notes,
            "mb": round(os.path.getsize(out_path) / 1048576, 2)}
