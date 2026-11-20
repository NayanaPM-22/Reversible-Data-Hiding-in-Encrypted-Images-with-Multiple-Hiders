import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    SetupRequest, HiderSecret, HidersEmbedRequest,
    MAX_HIDERS, MIN_HIDERS, MAX_IMAGE_DIM, MIN_IMAGE_DIM, MAX_SECRET_CHARS
)
from app.core.ass import ASS
from app.core.pehs import (
    pehs_embed_plaintext, pehs_extract_plaintext, psnr,
    estimate_capacity, text_to_bits, bits_to_text
)
from app.session_store import store

app = FastAPI(
    title="RDHEI-MH Demo",
    description=(
        "An implementation and interactive demo of the Additive Secret Sharing + "
        "Secure Multiparty Computation framework for Reversible Data Hiding in "
        "Encrypted Images with Multiple Hiders, based on Xiong et al. (2025), IEEE TDSC. "
        "Extended here to support full-color (RGB) images by running the paper's "
        "grayscale PEHS pipeline independently on each color channel."
    ),
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NUM_CHANNELS = 3  # R, G, B


def array_to_base64_png(arr):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.get("/api/about")
def about():
    return {
        "project": "RDHEI-MH Interactive Demo",
        "based_on": "Xiong, L., Zhou, W., Xia, Z., & Weng, J. (2025). A Universal Framework "
                    "for Reversible Data Hiding in Encrypted Images With Multiple Hiders. "
                    "IEEE TDSC, 22(4), 3944-3959.",
        "extension_note": "The published method operates on 8-bit grayscale images. This "
                           "implementation extends it to full-color images by running the "
                           "same validated pipeline independently on each RGB channel.",
        "implemented_by": ["Nayana Yogeshwari P M (231AI023)", "Nandeesh Nayak (231AI022)", "Lekhanashree (231AI017)"],
        "institution": "National Institute of Technology Karnataka, Surathkal, India",
    }


@app.get("/api/limits")
def get_limits():
    return {
        "min_hiders": MIN_HIDERS,
        "max_hiders": MAX_HIDERS,
        "min_image_dim": MIN_IMAGE_DIM,
        "max_image_dim": MAX_IMAGE_DIM,
        "max_secret_chars": MAX_SECRET_CHARS,
    }


@app.post("/api/session")
def create_session():
    session_id = store.create_session()
    return {"session_id": session_id}


@app.post("/api/setup")
def setup(req: SetupRequest):
    return {"status": "ok", "num_hiders": req.num_hiders, "limits": {
        "min_hiders": MIN_HIDERS, "max_hiders": MAX_HIDERS,
        "min_image_dim": MIN_IMAGE_DIM, "max_image_dim": MAX_IMAGE_DIM,
        "max_secret_chars": MAX_SECRET_CHARS,
    }}


@app.post("/api/owner/encrypt")
async def owner_encrypt(
    session_id: str = Form(...),
    num_hiders: int = Form(...),
    file: UploadFile = File(...),
):
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Unknown session. Please restart.")

    if num_hiders < MIN_HIDERS or num_hiders > MAX_HIDERS:
        raise HTTPException(status_code=422,
            detail=f"Number of hiders must be between {MIN_HIDERS} and {MAX_HIDERS}.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')   # <-- color, not grayscale
    except Exception:
        raise HTTPException(status_code=422, detail="File is not a valid image.")

    w, h = img.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        raise HTTPException(status_code=422,
            detail=f"Image too small ({w}x{h}). Minimum is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}.")
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        raise HTTPException(status_code=422,
            detail=f"Image too large ({w}x{h}). Maximum is {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}.")

    img = img.resize((512, 512))
    img_arr = np.array(img, dtype=np.int64)   # shape (512, 512, 3)

    # Compute capacity PER CHANNEL (each channel's own texture determines its own capacity)
    capacity_per_channel = [
        estimate_capacity(img_arr[:, :, c]) for c in range(NUM_CHANNELS)
    ]
    capacity_bits = sum(capacity_per_channel)

    if capacity_bits < 32:
        raise HTTPException(status_code=422,
            detail="This image doesn't have enough embedding capacity. Try a more textured image.")

    # ASS is purely elementwise math -- works identically on the full (512,512,3) color
    # array with no changes needed, unlike PEHS which needs true 2D per-channel structure.
    ass_demo = ASS(l=10, n=num_hiders)
    shares = ass_demo.share(img_arr)

    store.update(session_id,
        num_hiders=num_hiders,
        original_image=img_arr.tolist(),
        shares=[s.tolist() for s in shares],
        capacity_bits=capacity_bits,
        capacity_per_channel=capacity_per_channel,
    )

    share_previews = []
    for s in shares:
        s_f = np.array(s, dtype=np.float64)
        s_norm = 255 * (s_f - s_f.min()) / (s_f.max() - s_f.min() + 1e-9)
        share_previews.append(array_to_base64_png(s_norm))

    original_preview = array_to_base64_png(img_arr)

    return {
        "status": "ok",
        "capacity_bits": capacity_bits,
        "capacity_per_channel": capacity_per_channel,
        "capacity_chars_approx": max(0, (capacity_bits - 32) // 8),
        "share_previews": share_previews,
        "original_preview": original_preview,
    }


@app.post("/api/hiders/embed")
def hiders_embed(req: HidersEmbedRequest):
    session = store.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session. Please restart.")
    if 'shares' not in session:
        raise HTTPException(status_code=400, detail="Owner step not completed yet.")

    n = session['num_hiders']
    if len(req.secrets) != n:
        raise HTTPException(status_code=422,
            detail=f"Expected exactly {n} secrets (one per hider), got {len(req.secrets)}.")

    secrets_sorted = sorted(req.secrets, key=lambda s: s.hider_id)
    per_hider_bits = [text_to_bits(s.secret_text) for s in secrets_sorted]
    combined_bits = []
    for b in per_hider_bits:
        combined_bits.extend(b)

    capacity_bits = session['capacity_bits']
    if len(combined_bits) > capacity_bits:
        max_chars_total = max(0, (capacity_bits - 32 * n) // 8)
        raise HTTPException(status_code=422, detail=(
            f"Combined secrets need {len(combined_bits)} bits, but this image can only hold "
            f"{capacity_bits} bits. Reduce message length(s) -- roughly {max_chars_total} "
            f"total characters across all {n} hiders will fit."
        ))

    img_arr = np.array(session['original_image'], dtype=np.int64)   # (512,512,3)
    cap_per_channel = session['capacity_per_channel']

    # Slice combined_bits across channels in order: fill R's capacity first, then G, then B.
    # Extraction later concatenates in the SAME order, so this split is fully reversible.
    marked_channels = []
    ME_list, LE_list, n_embedded_list = [], [], []
    idx = 0
    for c in range(NUM_CHANNELS):
        channel_bits = combined_bits[idx: idx + cap_per_channel[c]]
        idx += len(channel_bits)
        marked_c, ME_c, LE_c, n_embedded_c = pehs_embed_plaintext(img_arr[:, :, c], channel_bits)
        marked_channels.append(marked_c)
        ME_list.append(ME_c)
        LE_list.append(LE_c)
        n_embedded_list.append(n_embedded_c)

    marked_arr = np.stack(marked_channels, axis=-1)   # back to (512,512,3)
    p = psnr(img_arr, marked_arr)

    ass_demo = ASS(l=10, n=n)
    marked_shares = ass_demo.share(marked_arr)
    marked_previews = []
    for s in marked_shares:
        s_f = np.array(s, dtype=np.float64)
        s_norm = 255 * (s_f - s_f.min()) / (s_f.max() - s_f.min() + 1e-9)
        marked_previews.append(array_to_base64_png(s_norm))

    marked_full_preview = array_to_base64_png(marked_arr.astype(np.uint8))

    store.update(req.session_id,
        marked_shares=[s.tolist() for s in marked_shares],
        ME_list=ME_list, LE_list=LE_list,
        chunk_sizes=[len(b) for b in per_hider_bits],
        secrets=[s.secret_text for s in secrets_sorted],
    )

    return {
        "status": "ok",
        "ME_per_channel": ME_list, "LE_per_channel": LE_list,
        "psnr_marked": p,
        "chunk_sizes": [len(b) for b in per_hider_bits],
        "marked_share_previews": marked_previews,
        "marked_full_preview": marked_full_preview,
    }


@app.post("/api/receiver/reconstruct")
def receiver_reconstruct(session_id: str = Form(...)):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session. Please restart.")
    if 'marked_shares' not in session:
        raise HTTPException(status_code=400, detail="Hiders step not completed yet.")

    n = session['num_hiders']
    ass_demo = ASS(l=10, n=n)
    marked_shares = [np.array(s, dtype=np.int64) for s in session['marked_shares']]
    marked_reconstructed = ass_demo.reconstruct(marked_shares)   # (512,512,3)

    ME_list, LE_list = session['ME_list'], session['LE_list']

    recovered_channels = []
    extracted_bits = []
    for c in range(NUM_CHANNELS):
        recovered_c, bits_c = pehs_extract_plaintext(
            marked_reconstructed[:, :, c], ME_list[c], LE_list[c])
        recovered_channels.append(recovered_c)
        extracted_bits.extend(bits_c)   # same R,G,B concatenation order as embedding

    recovered_arr = np.stack(recovered_channels, axis=-1)

    original = np.array(session['original_image'], dtype=np.int64)
    p_recovery = psnr(original, recovered_arr)

    chunk_sizes = session['chunk_sizes']
    secrets_out, idx = [], 0
    for size in chunk_sizes:
        chunk = extracted_bits[idx: idx + size]
        idx += size
        secrets_out.append(bits_to_text(chunk))

    all_match = secrets_out == session['secrets']

    return {
        "status": "ok",
        "extracted_secrets": secrets_out,
        "all_match": all_match,
        "psnr_recovery": None if p_recovery == float('inf') else p_recovery,
        "lossless": p_recovery == float('inf'),
        "recovered_image": array_to_base64_png(recovered_arr.astype(np.uint8)),
    }