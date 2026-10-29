# ComfyUI-MotionDeck

Slideshow-style compositor for Graydient: up to 8 videos + 8 stills + a scene manifest -> one mp4,
with grid-reveal zooms, splits, camera moves, animated titles, transitions and an optional music bed.
Pure PIL + ffmpeg (no model). Sub-pixel box resizes, so camera moves do not jitter like ffmpeg zoompan.

## Manifest

Scenes start with the word `scene`; `title:` and `sub:` run to the end of the scene.
`anim NAME` picks the title entrance: `rise` (default), `pop`, `slide`, `type`, `wipe`, `zoom`, `drop`, `blur`, or `cycle` to vary scene by scene.

    scene grid i1 v1 i2 i3 reveal hold 2 title: My Trip sub: summer
    scene single i4 push fade 0.6 title: Close up
    scene split v2 v3 slide 0.5 hold 3
    scene card hold 3 wipe title: The End

`i1..i8` stills, `v1..v8` clips. Layouts `grid|split|single|card`; grid modes `reveal` (zoom into each panel
in turn, the clip plays only while zoomed in) or `static`; `hold N`, `cols N`; single cameras
`push|pull|panleft|panright|drift`; transitions `fade|cut|slide|wipe [seconds]`. An empty manifest makes
four-panel reveal scenes from whatever media is attached.

## Inputs
`width`/`height` (0 = 1280x720, snapped to 16), `fps`, `max_frames`, `style` 0-3, `pace` 0.5-2, optional audio.
Measured offline: ~39 ms/frame at 1280x720 (about real time), so keep `max_frames` <= 1800 on the 180 s budget.

## Speaker scenes (interview -> vertical, word captions)

    scene speaker v1 hold 60 caps karaoke font 1 size 6 hl yellow title: Hook line

Transcribes the clip with word timing (faster-whisper, audio passed as an array), finds faces (OpenCV YuNet, Haar
fallback), tracks up to 3 people by x position and picks the speaker per word from mouth-region motion while that
word is spoken, with a 1.4 s minimum dwell. Output is a hard-cut crop to the speaker (default 720x1280). The source
audio is kept (and any `audio_file` bed is ducked under it).

Options: `start N`, `hold N`, `stack` (both speakers, one above the other), `fixed N` (lock onto person N),
`lang xx`, `whisper <model>`; captions `caps karaoke|single|off`, `font 0-3` (DejaVu, Anton, Bebas, Poppins),
`size N` (% of frame height), `words N`, `capspos N`, `hl`/`fg` colour, `stroke N`, `case upper|lower|normal`.

Clean-room implementation: no code from ClipsKitty (AGPL-3.0). Fonts Anton, Bebas Neue and Poppins are SIL OFL.
Model files expected under `models/`: `whisper/large-v3/*` and `yunet/face_detection_yunet_2023mar.onnx` (both are
also fetched at run time if missing).
