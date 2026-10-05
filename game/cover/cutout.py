"""Cut the white background out of a sticker image -> RGBA PNG.

    python game/cover/cutout.py game/cover/paimon_src.jpg game/cover/paimon.png

Background = near-white pixels connected to the image border (white hair / clothes are
enclosed by dark outlines, so they stay). Edge pixels next to the background get a soft
alpha from their lightness, so the anti-aliased outline blends into any backdrop.
"""
import sys, numpy as np
from PIL import Image
from scipy import ndimage

TOP_BAND = 260     # px: rows above the head (halo); white regions enclosed only there are background

def main(src, dst):
    im = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
    light = im.min(-1)
    white = light > 232
    lab, n = ndimage.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    # enclosed white areas entirely in the top band (inside the halo ring) are background too
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if i not in border and sl is not None and sl[0].stop < TOP_BAND: bg |= lab == i
    alpha = np.where(bg, 0.0, 1.0)
    # soft edge: within 3 px of the background, alpha from how dark the pixel is
    near = ndimage.binary_dilation(bg, iterations=3) & ~bg
    soft = np.clip((255 - light) / (255 - 150), 0, 1)
    alpha[near] = np.maximum(soft[near], 0.0)
    alpha = ndimage.gaussian_filter(alpha, 0.6) * (1 - bg * 0.0)
    alpha[bg & ~ndimage.binary_dilation(~bg, iterations=1)] = 0
    # un-blend the white from soft pixels so the edge does not glow on other colours
    a = np.clip(alpha, 1e-3, 1)[..., None]
    rgb = np.clip((im - 255 * (1 - a)) / a, 0, 255)
    rgb = np.where(alpha[..., None] > 0.98, im, rgb)
    out = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    Image.fromarray(out).save(dst)
    print(dst, f"background {bg.mean() * 100:.1f}% of pixels")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
