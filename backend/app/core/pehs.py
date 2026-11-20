import numpy as np

def sgap_predict(img):
    """Vectorized SGAP prediction (fast, no Python-level pixel loops)."""
    img = img.astype(np.int64)
    h, w = img.shape
    pred = np.zeros_like(img)
    valid_mask = np.zeros((h, w), dtype=bool)

    N  = img[0:h-1, 1:w-1]
    W  = img[1:h,   0:w-2]
    NE = img[0:h-1, 2:w]
    NW = img[0:h-1, 0:w-2]

    pred_region = ((N + W) >> 1) + ((NE - NW) >> 2)

    pred[1:h, 1:w-1] = pred_region
    valid_mask[1:h, 1:w-1] = True

    return pred, valid_mask

def sgap_predict_single(arr, i, j):
    N = arr[i - 1, j]; W = arr[i, j - 1]
    NE = arr[i - 1, j + 1]; NW = arr[i - 1, j - 1]
    return ((N + W) >> 1) + ((NE - NW) >> 2)

def estimate_capacity(img_arr):
    """Max embeddable bits for this specific image (pixel count at histogram peak)."""
    pred, valid_mask = sgap_predict(img_arr)
    e = img_arr.astype(np.int64) - pred
    abs_errors = np.abs(e[valid_mask])
    if len(abs_errors) == 0:
        return 0
    max_e = abs_errors.max()
    hist = np.bincount(abs_errors, minlength=max_e + 1)
    ME = int(np.argmax(hist))
    return int(hist[ME])

def pehs_embed_plaintext(img, data_bits):
    img = img.astype(np.int64)
    h, w = img.shape
    pred_full, valid_mask = sgap_predict(img)
    e_full = img - pred_full
    abs_errors = np.abs(e_full[valid_mask])
    max_e = abs_errors.max()
    hist = np.bincount(abs_errors, minlength=max_e + 1)
    ME = int(np.argmax(hist))
    LE = None
    for candidate in range(ME + 1, max_e + 2):
        if candidate > max_e or hist[candidate] == 0:
            LE = candidate; break
    if LE is None:
        LE = max_e + 1

    marked = img.copy()
    bit_idx = 0
    bits_embedded = 0
    data_bits = list(data_bits)

    for i in range(1, h):
        for j in range(1, w - 1):
            err = e_full[i, j]
            abs_e = abs(err)
            sign = 1 if err >= 0 else -1
            if abs_e == ME:
                if bit_idx < len(data_bits):
                    b = data_bits[bit_idx]; bit_idx += 1; bits_embedded += 1
                    marked[i, j] = img[i, j] + sign * b
            elif ME < abs_e < LE:
                marked[i, j] = img[i, j] + sign * 1

    return marked, ME, LE, bits_embedded

def pehs_extract_plaintext(marked_img, ME, LE):
    marked_img = marked_img.astype(np.int64)
    h, w = marked_img.shape
    recovered = marked_img.copy()
    extracted_bits = []
    for i in range(1, h):
        for j in range(1, w - 1):
            pred = sgap_predict_single(recovered, i, j)
            err = recovered[i, j] - pred
            abs_e = abs(err)
            sign = 1 if err >= 0 else -1
            if abs_e == ME:
                extracted_bits.append(0)
            elif abs_e == ME + 1:
                extracted_bits.append(1)
                recovered[i, j] = recovered[i, j] - sign * 1
            elif ME + 1 < abs_e <= LE:
                recovered[i, j] = recovered[i, j] - sign * 1
    return recovered, extracted_bits

def psnr(orig, test):
    mse = np.mean((orig.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

def text_to_bits(text):
    encoded = text.encode('utf-8')
    length = len(encoded)
    length_bits = [int(b) for b in format(length, '032b')]
    data_bits = []
    for byte in encoded:
        data_bits.extend([int(b) for b in format(byte, '08b')])
    return length_bits + data_bits

def bits_to_text(bits):
    length_bits = bits[:32]
    length = int(''.join(map(str, length_bits)), 2)
    data_bits = bits[32:32 + length * 8]
    byte_vals = []
    for k in range(0, len(data_bits), 8):
        byte_str = ''.join(map(str, data_bits[k:k+8]))
        byte_vals.append(int(byte_str, 2))
    return bytes(byte_vals).decode('utf-8', errors='replace')