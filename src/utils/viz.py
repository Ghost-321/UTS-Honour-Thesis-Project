import numpy as np, cv2
from PIL import Image, ImageDraw

def overlay_mask_rgb(img: np.ndarray, mask: np.ndarray, color=(255,0,0), alpha=0.35):
    if img.dtype != np.uint8:
        img_vis = (np.clip(img,0,1)*255).astype(np.uint8)
    else:
        img_vis = img.copy()
    if mask.dtype != np.uint8:
        mask = (mask>0).astype(np.uint8)
    overlay = np.zeros_like(img_vis)
    overlay[mask>0] = color
    return cv2.addWeighted(img_vis, 1.0, overlay, alpha, 0)

def pil_from_float(img: np.ndarray):
    return Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))

def montage(images, titles=None, cols=3, pad=8):
    ims = [(im if im.dtype==np.uint8 else (np.clip(im,0,1)*255).astype(np.uint8)) for im in images]
    h = max(im.shape[0] for im in ims); w = max(im.shape[1] for im in ims)
    rows = (len(ims)+cols-1)//cols
    canvas = Image.new("RGB", (cols*w + (cols+1)*pad, rows*h + (rows+1)*pad), color=(30,30,30))
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        y = r*h + (r+1)*pad; x = c*w + (c+1)*pad
        tile = Image.fromarray(cv2.resize(im, (w,h), interpolation=cv2.INTER_AREA))
        canvas.paste(tile, (x,y))
        if titles and i < len(titles):
            draw.text((x+6, y+6), titles[i], fill=(255,255,255))
    return canvas
