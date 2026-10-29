"""Motion Deck: manifest -> scenes (grid-reveal zoom, split, single, title card) -> ffmpeg mp4.

No ComfyUI imports; tested offline. Frames are composed with PIL sub-pixel box resizes (no ffmpeg
zoompan, which jitters) and streamed to ffmpeg's stdin.
"""
import math
import os
import re
import shutil
import subprocess
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

UA = {"User-Agent": "Mozilla/5.0 (MotionDeck)"}
IMG_HOLD = 2.5       # default seconds a still panel / scene is held
VID_HOLD_MAX = 6.0   # default cap for a video panel's hold
MOVE = 0.9           # seconds for a camera move between wide view and a panel
WIDE_HOLD = 0.7
FADE_DEFAULT = 0.5
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


def _num(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


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
        toks = head.replace(",", " ").split()
        low = [t.lower() for t in toks]
        i = 0
        while i < len(low):
            t = low[i]
            nxt = _num(low[i + 1]) if i + 1 < len(low) else None
            m = re.fullmatch(r"([iv])([1-8])", t)
            if m:
                sc.media.append((m.group(1), int(m.group(2))))
            elif t in ("grid", "split", "single", "card"):
                sc.layout = t
            elif t in ("reveal", "static"):
                sc.mode = t
            elif t == "hold" and nxt is not None:
                sc.hold = max(0.5, min(30.0, nxt))
                i += 1
            elif t == "cols" and nxt is not None:
                sc.cols = max(1, min(4, int(nxt)))
                i += 1
            elif t in ("fade", "cut", "slide", "wipe"):
                sc.trans = t
                if nxt is not None:
                    sc.trans_dur = max(0.1, min(3.0, nxt))
                    i += 1
            elif t in ("push", "pull", "panleft", "panright", "drift"):
                sc.cam = t
            else:
                notes.append(f"ignored word '{toks[i]}'")
            i += 1
        scenes.append(sc)
    if not scenes:    # no manifest: groups of 4 as grid-reveal scenes
        media = [("i", i) for i in range(1, n_images + 1)] + [("v", i) for i in range(1, n_videos + 1)]
        for k in range(0, len(media), 4):
            sc = Scene()
            sc.media = media[k:k + 4]
            scenes.append(sc)
    for sc in scenes:
        if sc.layout is None:
            sc.layout = "card" if not sc.media else ("single" if len(sc.media) == 1 else "grid")
        if sc.layout == "single":
            sc.media = sc.media[:1]
        if sc.layout == "split":
            sc.media = sc.media[:2]
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


def text_layer(title, sub, W, H, style, card=False):
    """RGBA full-frame layer with the title (and subtitle), centred (card) or lower-third (over media)."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tsize = max(24, int(H * (0.11 if card else 0.075)))
    ssize = max(16, int(tsize * 0.45))
    tf, sf = _font(tsize), _font(ssize, bold=False)
    tl = _wrap(d, title, tf, int(W * 0.84)) if title else []
    sl = _wrap(d, sub, sf, int(W * 0.8)) if sub else []
    th = int(tsize * 1.2) * len(tl) + (int(ssize * 1.4) * len(sl) + int(ssize * 0.6) if sl else 0)
    y = (H - th) // 2 if card else int(H * 0.82) - th
    if not card and tl:    # soft band behind the text so it reads over any picture
        band = Image.new("RGBA", (W, th + int(tsize * 0.8)), (0, 0, 0, 120))
        layer.alpha_composite(band, (0, max(0, y - int(tsize * 0.4))))
        d = ImageDraw.Draw(layer)
    stroke = max(2, tsize // 14)
    fill, sk = style["title"], style["stroke"]
    for ln in tl:
        d.text(((W - d.textlength(ln, font=tf)) / 2, y), ln, font=tf, fill=fill, stroke_width=stroke, stroke_fill=sk)
        y += int(tsize * 1.2)
    if sl:
        y += int(ssize * 0.6)
        for ln in sl:
            d.text(((W - d.textlength(ln, font=sf)) / 2, y), ln, font=sf, fill=style["accent"],
                   stroke_width=max(1, ssize // 14), stroke_fill=sk)
            y += int(ssize * 1.4)
    return layer


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
            self.layer = text_layer(sc.title, sc.sub, W, H, style, card=True)
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
        self.layer = text_layer(sc.title, sc.sub, W, H, style) if (sc.title or sc.sub) else None
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
            img = Image.new("RGB", (self.W, self.H), self.style["bg"])
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
        t0, fin, fout = 0.3, 0.45, 0.35
        end = min(dur, t0 + (self.title_dur if self.kind != "card" else self.duration - 0.6))
        if t < t0 or t > end:
            return
        a = min(1.0, (t - t0) / fin) * min(1.0, (end - t) / fout)
        a = max(0.0, a)
        rise = int((1.0 - _ease((t - t0) / fin)) * self.H * 0.03)
        layer = self.layer
        if a < 1.0:
            layer = layer.copy()
            layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
        img.paste(layer, (0, rise), layer)

    def close(self):
        for po in self.panel_objs:
            if po.reader is not None:
                po.reader.close()


# ── transitions ───────────────────────────────────────────────────────────────

def blend(a, b, p, kind):
    p = _ease(p)
    if kind == "slide":
        W, H = a.size
        off = int(W * p)
        out = Image.new("RGB", (W, H))
        out.paste(a, (-off, 0))
        out.paste(b, (W - off, 0))
        return out
    if kind == "wipe":
        W, H = a.size
        x = int(W * p)
        out = a.copy()
        out.paste(b.crop((0, 0, x, H)), (0, 0))
        return out
    return Image.blend(a, b, p)


# ── compile ───────────────────────────────────────────────────────────────────

def compile_deck(manifest, vid_files, img_files, audio_src, out_path, work, input_dir,
                 width=0, height=0, fps=24, max_frames=2400, style_id=0, pace=1.0):
    W = int(width) or 0
    H = int(height) or 0
    if not W and not H:
        W, H = 1280, 720
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
    for sc in scenes:
        miss = [f"{k}{i}" for k, i in sc.media if (k, i) not in paths]
        if miss:
            notes.append(f"missing media {', '.join(miss)} (skipped)")
    renders = []
    try:
        for sc in scenes:
            renders.append(SceneRender(sc, paths, W, H, fps, style, pace))
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
        if audio:
            cmd += ["-i", audio]
        cmd += ["-map", "0:v:0"]
        if audio:
            secs = total / fps
            cmd += ["-map", "1:a:0", "-af", f"afade=t=out:st={max(0.0, secs - 1.5):.2f}:d=1.5", "-c:a", "aac",
                    "-b:a", "160k", "-t", f"{secs:.3f}"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out_path]
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
    finally:
        for r in renders:
            r.close()
    return {"size": f"{W}x{H}", "fps": fps, "frames": total, "seconds": round(total / fps, 2),
            "scenes": [{"layout": r.sc.layout, "mode": r.sc.mode, "panels": len(r.panels),
                        "seconds": round(r.duration, 2), "title": r.sc.title[:60]} for r in renders],
            "images": n_img, "videos": n_vid, "notes": notes,
            "mb": round(os.path.getsize(out_path) / 1048576, 2)}
