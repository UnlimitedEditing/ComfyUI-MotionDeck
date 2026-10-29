import asyncio
import json
import os
import tempfile

from . import core


def _encode_string_as_image(text):
    """Lossless data image: 4-byte BE length + UTF-8, 3 bytes/pixel (ForgeExpress convention)."""
    import struct
    import numpy as np
    import torch
    raw = text.encode("utf-8")
    payload = struct.pack(">I", len(raw)) + raw
    while len(payload) % 3:
        payload += bytes(1)
    side = max(1, int(np.ceil(np.sqrt(len(payload) // 3))))
    payload += bytes(1) * (side * side * 3 - len(payload))
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(side, side, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class MotionDeckCompile:
    """Slideshow / grid-reveal compositor: up to 8 videos + 8 stills + a scene manifest -> one mp4."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "manifest": ("STRING", {"multiline": True, "default": ""}),
            "width": ("INT", {"default": 0, "min": 0, "max": 1920, "step": 16}),
            "height": ("INT", {"default": 0, "min": 0, "max": 1920, "step": 16}),
            "fps": ("INT", {"default": 24, "min": 12, "max": 60}),
            "max_frames": ("INT", {"default": 1800, "min": 48, "max": 7200}),
            "style": ("INT", {"default": 0, "min": 0, "max": 3}),
            "pace": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            "audio_file": ("STRING", {"default": ""}),
        }
        for i in range(1, 9):
            req[f"video{i}"] = ("STRING", {"default": ""})
        for i in range(1, 9):
            req[f"image{i}"] = ("STRING", {"default": ""})
        req["audio_init_filename"] = ("STRING", {"default": ""})
        req["audio_init_url"] = ("STRING", {"default": ""})
        req["video_init_filename"] = ("STRING", {"default": ""})
        req["video_init_url"] = ("STRING", {"default": ""})
        return {"required": req}

    RETURN_TYPES = ("VIDEO", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "report", "report_image")
    FUNCTION = "run"
    CATEGORY = "MotionDeck"

    async def run(self, manifest, width, height, fps, max_frames, style, pace, audio_file,
                  video1, video2, video3, video4, video5, video6, video7, video8,
                  image1, image2, image3, image4, image5, image6, image7, image8,
                  audio_init_filename, audio_init_url, video_init_filename, video_init_url):
        # blocking PIL/ffmpeg work stays off the event loop (a blocked loop can make a finished job report timed_out)
        return await asyncio.to_thread(
            self._run_sync, manifest, width, height, fps, max_frames, style, pace,
            audio_file or audio_init_filename or audio_init_url,
            [video1 or video_init_filename or video_init_url, video2, video3, video4, video5, video6, video7, video8],
            [image1, image2, image3, image4, image5, image6, image7, image8])

    def _run_sync(self, manifest, width, height, fps, max_frames, style, pace, audio_src, vids, imgs):
        from comfy_api.latest import InputImpl
        try:
            import folder_paths
            input_dir = folder_paths.get_input_directory()
        except Exception:
            input_dir = ""
        work = tempfile.mkdtemp(prefix="motiondeck_")
        out = os.path.join(work, "deck.mp4")
        report = core.compile_deck(manifest, vids, imgs, audio_src, out, work, input_dir, width=width,
                                   height=height, fps=fps, max_frames=max_frames, style_id=style, pace=pace)
        text = json.dumps(report, indent=1)
        return (InputImpl.VideoFromFile(out), text, _encode_string_as_image(text))


NODE_CLASS_MAPPINGS = {"MotionDeckCompile": MotionDeckCompile}
NODE_DISPLAY_NAME_MAPPINGS = {"MotionDeckCompile": "Motion Deck (grid-reveal slideshow compositor)"}
