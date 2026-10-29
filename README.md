# ComfyUI-MotionDeck

Slideshow-style compositor for Graydient: up to 8 videos + 8 stills + a scene manifest -> one mp4,
with grid-reveal zooms, splits, camera moves, animated titles, transitions and an optional music bed.
Pure PIL + ffmpeg (no model). Sub-pixel box resizes, so camera moves do not jitter like ffmpeg zoompan.

## Manifest

Scenes start with the word `scene`; `title:` and `sub:` run to the end of the scene.

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
