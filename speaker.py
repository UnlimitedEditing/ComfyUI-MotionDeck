"""Speaker-follow vertical crop + per-word captions (clean-room: no ClipsKitty code, AGPL).

Pipeline for a `scene speaker` source clip:
  1. ffmpeg -> 16 kHz mono wav -> faster-whisper word timestamps (an ARRAY is passed, never a path:
     Graydient's PyAV breaks faster-whisper's own decoder, KI-007 s17).
  2. 10 fps face sampling with OpenCV YuNet (Haar fallback), faces tracked into identities by x position,
     mouth-region motion per identity.
  3. Speaker per word = identity with the most (normalised) mouth motion while the word is spoken, with a
     minimum dwell so the cut does not flicker.
  4. Crop plan (hard cuts between speakers) + caption groups.
cv2 is part of the ComfyUI base image (KI-007), so no pip entry is needed.
"""
import os
import subprocess
import urllib.request
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FONTS = ["DejaVuSans-Bold.ttf", "Anton-Regular.ttf", "BebasNeue-Regular.ttf", "Poppins-ExtraBold.ttf"]
COLORS = {"white": (255, 255, 255), "yellow": (255, 221, 0), "green": (70, 235, 120), "pink": (255, 90, 190),
          "cyan": (60, 230, 240), "orange": (255, 150, 30), "red": (255, 70, 60), "black": (0, 0, 0),
          "blue": (80, 140, 255), "purple": (170, 100, 255)}
_WHISPER = {}


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def probe_video(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return "audio" in r.stdout


def _models_dir():
    try:
        import folder_paths
        return folder_paths.models_dir
    except Exception:
        return os.environ.get("MOTIONDECK_MODELS", "")


# ── 1. words ──────────────────────────────────────────────────────────────────

def decode_wav16k(src, start, dur, work):
    wav = os.path.join(work, "speech16k.wav")
    r = _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src, "-vn", "-ac", "1",
              "-ar", "16000", "-c:a", "pcm_s16le", wav])
    if r.returncode or not os.path.isfile(wav):
        return None
    with wave.open(wav, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_words(samples, language="", model_name=None):
    """Word list [{start, end, word}] from faster-whisper. The model loads from models/whisper/<name>
    when staged by concept_mapping, else from the hub."""
    from faster_whisper import WhisperModel
    name = model_name or os.environ.get("MOTIONDECK_WHISPER", "large-v3")
    key = name
    if key not in _WHISPER:
        local = os.path.join(_models_dir(), "whisper", name)
        load_from = local if os.path.isfile(os.path.join(local, "model.bin")) else name
        try:
            import torch
            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        _WHISPER[key] = WhisperModel(load_from, device="cuda" if cuda else "cpu",
                                     compute_type="float16" if cuda else "int8")
    segs, _ = _WHISPER[key].transcribe(samples, language=language or None, word_timestamps=True,
                                       vad_filter=True, beam_size=3)
    words = []
    for s in segs:
        for w in (s.words or []):
            t = (w.word or "").strip()
            if t:
                words.append({"start": float(w.start), "end": max(float(w.end), float(w.start) + 0.05), "word": t})
    return words


# ── 2. faces + mouth motion ───────────────────────────────────────────────────

def _detector():
    import cv2
    path = os.path.join(_models_dir(), "yunet", "face_detection_yunet_2023mar.onnx")
    if not os.path.isfile(path):
        cache = os.path.join(os.environ.get("TEMP", "/tmp"), "motiondeck_yunet.onnx")
        if not os.path.isfile(cache):
            try:
                urllib.request.urlretrieve(YUNET_URL, cache)
            except Exception:
                cache = ""
        path = cache
    if path and hasattr(cv2, "FaceDetectorYN"):
        det = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.6, 0.3, 200)
        return "yunet", det
    return "haar", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def sample_faces(src, start, dur, fps=10, width=640):
    """Returns (times, per-frame detections, size). Detection = dict(cx, cy, w, mouth=(x,y,hw), gray patch)."""
    import cv2
    sw0, sh0 = probe_video(src)
    sw = width
    sh = int(round(sh0 * width / sw0 / 2)) * 2
    kind, det = _detector()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
                          "-vf", f"fps={fps},scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    fsz = sw * sh * 3
    while True:
        buf = p.stdout.read(fsz)
        if len(buf) < fsz:
            break
        img = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = []
        if kind == "yunet":
            det.setInputSize((sw, sh))
            _, fc = det.detect(img)
            for r in (fc if fc is not None else []):
                x, y, w, h = r[:4]
                mx = (r[10] + r[12]) / 2.0
                my = (r[11] + r[13]) / 2.0
                hw = max(6.0, abs(r[12] - r[10]) * 0.9)
                dets.append({"cx": x + w / 2, "cy": y + h / 2, "w": w, "score": float(r[14]),
                             "mouth": _patch(gray, mx, my, hw)})
        else:
            for (x, y, w, h) in det.detectMultiScale(gray, 1.2, 5, minSize=(40, 40)):
                dets.append({"cx": x + w / 2, "cy": y + h / 2, "w": w, "score": 1.0,
                             "mouth": _patch(gray, x + w / 2, y + 0.78 * h, w * 0.3)})
        frames.append(dets)
    p.stdout.close()
    p.wait()
    times = np.arange(len(frames)) / float(fps)
    return times, frames, (sw, sh)


def _patch(gray, cx, cy, hw):
    import cv2
    h, w = gray.shape
    hh = hw * 0.65
    x0, x1 = int(max(0, cx - hw)), int(min(w, cx + hw))
    y0, y1 = int(max(0, cy - hh)), int(min(h, cy + hh))
    if x1 - x0 < 4 or y1 - y0 < 3:
        return None
    pt = cv2.resize(gray[y0:y1, x0:x1], (24, 14), interpolation=cv2.INTER_AREA).astype(np.float32)
    return pt - pt.mean()


def build_tracks(times, frames, size):
    """Assign detections to up to 3 identities by x position. Returns dict(K, x[K,T] (0-1), act[K,T])."""
    T = len(frames)
    counts = [len(f) for f in frames]
    if T == 0 or max(counts) == 0:
        return {"K": 0, "x": np.zeros((0, 0)), "act": np.zeros((0, 0)), "times": times}
    K = int(min(3, max(1, np.percentile([c for c in counts if c > 0], 80))))
    rows = [sorted(d["cx"] for d in f) for f in frames if len(f) == K]
    if rows:
        centers = np.mean(np.array(rows), axis=0)
    else:
        xs = sorted(d["cx"] for f in frames for d in f)
        centers = np.array([xs[int((i + 0.5) * len(xs) / K)] for i in range(K)])
    x = np.full((K, T), np.nan)
    act = np.full((K, T), np.nan)
    prev = [None] * K
    for t, f in enumerate(frames):
        taken = set()
        for d in sorted(f, key=lambda d: -d["score"]):
            k = int(np.argmin(np.abs(centers - d["cx"])))
            if k in taken:
                continue
            taken.add(k)
            x[k, t] = d["cx"] / size[0]
            m = d["mouth"]
            if m is not None and prev[k] is not None:
                act[k, t] = float(np.mean(np.abs(m - prev[k])))
            prev[k] = m
        for k in range(K):
            if k not in taken:
                prev[k] = None
    for k in range(K):                 # fill gaps so a missed detection does not move the crop
        for arr in (x[k], act[k]):
            good = ~np.isnan(arr)
            if good.any():
                arr[:] = np.interp(np.arange(T), np.flatnonzero(good), arr[good])
            else:
                arr[:] = 0.5 if arr is x[k] else 0.0
        x[k] = _smooth(x[k], 15)
    return {"K": K, "x": x, "act": act, "times": times}


def _smooth(a, n):
    if len(a) < 3:
        return a
    n = max(1, min(n, len(a)))
    pad = np.pad(a, (n // 2, n - n // 2 - 1), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


# ── 3. who speaks ─────────────────────────────────────────────────────────────

def decide_speakers(words, tracks, dur, min_dwell=1.4, lead=0.12, fixed=None):
    """Returns segments [(t0, t1, identity)] covering 0..dur."""
    K = tracks["K"]
    if K <= 1 or not words:
        return [(0.0, dur, 0)]
    if fixed is not None:
        return [(0.0, dur, min(max(fixed, 0), K - 1))]
    times = tracks["times"]
    act = np.array([_smooth(tracks["act"][k], 5) for k in range(K)])
    norm = np.array([a / (np.median(a[a > 0]) + 1e-6) if (a > 0).any() else a for a in act])
    who = []
    last = 0
    for w in words:
        m = (times >= w["start"] - 0.05) & (times <= w["end"] + 0.05)
        if m.any():
            sc = norm[:, m].mean(axis=1)
            if sc.max() > 0 and (np.sort(sc)[-1] - np.sort(sc)[-2]) > 0.02:
                last = int(np.argmax(sc))
        who.append(last)
    segs = []                           # merge words into runs
    for w, s in zip(words, who):
        if segs and segs[-1][2] == s:
            segs[-1][1] = w["end"]
        else:
            segs.append([w["start"], w["end"], s])
    merged = True                      # absorb runs shorter than the minimum dwell into the previous speaker
    while merged and len(segs) > 1:
        merged = False
        for i in range(len(segs)):
            if segs[i][1] - segs[i][0] < min_dwell:
                j = i - 1 if i > 0 else i + 1
                if segs[j][2] != segs[i][2] or True:
                    a, b = (j, i) if j < i else (i, j)
                    segs[a] = [segs[a][0], segs[b][1], segs[j][2]]
                    del segs[b]
                    merged = True
                    break
        for i in range(len(segs) - 1):           # join neighbours that became the same speaker
            if segs[i][2] == segs[i + 1][2]:
                segs[i] = [segs[i][0], segs[i + 1][1], segs[i][2]]
                del segs[i + 1]
                merged = True
                break
    out = []
    for i, (a, b, s) in enumerate(segs):
        t0 = 0.0 if i == 0 else max(0.0, a - lead)
        t1 = dur if i == len(segs) - 1 else max(0.0, segs[i + 1][0] - lead)
        out.append((t0, t1, s))
    return out


# ── 4. captions ───────────────────────────────────────────────────────────────

def _font(idx, size):
    p = os.path.join(os.path.dirname(__file__), "fonts", FONTS[int(idx) % len(FONTS)])
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


class Captions:
    """Word-timed captions. style 'karaoke': a few words visible, the spoken one highlighted;
    'single': one word at a time. Layers are pre-rendered per (group, active word) and cached."""

    def __init__(self, words, W, H, opts):
        self.W, self.H = W, H
        self.style = opts.get("caps", "karaoke")
        self.font = opts.get("font", 0)
        self.size = max(16, int(H * float(opts.get("size", 5.0)) / 100.0))
        self.fg = COLORS.get(opts.get("fg", "white"), (255, 255, 255))
        self.hl = COLORS.get(opts.get("hl", "yellow"), (255, 221, 0))
        self.stroke = int(opts.get("stroke", max(2, self.size // 9)))
        self.case = opts.get("case", "upper")
        self.ypos = float(opts.get("capspos", 68.0)) / 100.0
        n = int(opts.get("words", 3))
        self.words = []
        for w in words:
            t = w["word"]
            t = t.upper() if self.case == "upper" else t.lower() if self.case == "lower" else t
            self.words.append(dict(w, word=t))
        self.groups = []
        cur = []
        for i, w in enumerate(self.words):
            cur.append(i)
            gap = (self.words[i + 1]["start"] - w["end"]) if i + 1 < len(self.words) else 9
            if len(cur) >= n or w["word"][-1:] in ".!?;:" or gap > 0.7:
                self.groups.append(cur)
                cur = []
        if cur:
            self.groups.append(cur)
        self.cache = {}
        self.starts = [self.words[g[0]]["start"] for g in self.groups]

    def _layout(self, gi, active):
        key = (gi, active)
        if key in self.cache:
            return self.cache[key]
        idxs = self.groups[gi]
        if self.style == "single":
            idxs = [active]
        d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        pop = 1.12
        items = []
        for i in idxs:
            f = _font(self.font, int(self.size * (pop if i == active else 1.0)))
            items.append((self.words[i]["word"], f, i == active))
        space = self.size * 0.28
        max_w = self.W * 0.9
        lines, cur, cw = [], [], 0.0
        for it in items:
            tw = d.textlength(it[0], font=it[1])
            if cur and cw + space + tw > max_w:
                lines.append((cur, cw))
                cur, cw = [], 0.0
            cur.append((it, tw))
            cw += tw + (space if len(cur) > 1 else 0)
        if cur:
            lines.append((cur, cw))
        lh = int(self.size * 1.35)
        pad = self.stroke + 6
        img = Image.new("RGBA", (int(self.W), lh * len(lines) + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for li, (cur_, cw_) in enumerate(lines):
            x = (self.W - cw_) / 2
            for (txt, f, act), tw in cur_:
                d.text((x, pad + li * lh), txt, font=f, fill=(self.hl if act else self.fg),
                       stroke_width=self.stroke, stroke_fill=(0, 0, 0))
                x += tw + space
        self.cache[key] = img
        return img

    def draw(self, frame, t):
        """Composite the caption for time t onto an RGB PIL frame (in place)."""
        if not self.groups:
            return
        gi = None
        for i, g in enumerate(self.groups):
            end = self.words[g[-1]]["end"] + 0.18
            if self.words[g[0]]["start"] - 0.05 <= t <= end:
                gi = i
                break
        if gi is None:
            return
        g = self.groups[gi]
        active = g[0]
        for j, wi in enumerate(g):
            nxt = self.words[g[j + 1]]["start"] if j + 1 < len(g) else self.words[wi]["end"] + 0.18
            if self.words[wi]["start"] - 0.02 <= t < nxt:
                active = wi
                break
        layer = self._layout(gi, active)
        y = int(self.H * self.ypos - layer.height / 2)
        frame.paste(layer, (0, max(0, min(self.H - layer.height, y))), layer)
