"""Exploratory (12 November 2026), run as: PYTHONPATH=workflows <TF 2.14.1 python> workflows/abft_pair_count_explore.py
Exploratory: share of double faults per ABFT channel that end in a wrong repair (pinned ResNet-8 stored codes).
Checksums per output channel: C1 = sum_i w_i, C2 = sum_i i*w_i, i = 1..G, exact integers.  Decoder: syndromes
s1 = sum(errors), s2 = sum(i*errors).  s1 == 0 -> no repair.  s1 != 0 and s2/s1 an integer l in 1..G -> repair
weight l by subtracting s1.  Otherwise detected, left as is."""
import numpy as np
from runP_phase0 import Model

def deltas(codes):
    c = codes.astype(np.int16)
    u = c.astype(np.uint8).astype(np.int16)
    out = np.empty((c.size, 8), dtype=np.int64)
    for a in range(8):
        f = (u ^ (1 << a)).astype(np.uint8).view(np.int8).astype(np.int64)
        out[:, a] = f - c
    return out

m = Model("pinned", 4)
tot = dict(pairs=0, cancel=0, wrong=0, detected=0, ck_pairs=0, ck_wrong=0)
rows = []
for i, n in enumerate(m.names):
    v = m.codes[m.starts[i]: m.starts[i + 1]].reshape(m.shapes[n][0], -1)
    G = v.shape[1]
    lay = dict(pairs=0, cancel=0, wrong=0, ck_pairs=0, ck_wrong=0)
    for ch in v:
        d = deltas(ch).reshape(-1)                      # 8G single-bit errors
        idx = np.repeat(np.arange(1, G + 1), 8)
        I, K = np.triu_indices(d.size, 1)
        diff = idx[I] != idx[K]
        I, K = I[diff], K[diff]
        s1 = d[I] + d[K]; s2 = idx[I] * d[I] + idx[K] * d[K]
        nz = s1 != 0
        ok = np.zeros_like(nz); ok[nz] = (s2[nz] % s1[nz] == 0)
        l = np.zeros_like(s1); l[ok] = s2[ok] // s1[ok]
        wrong = ok & (l >= 1) & (l <= G)
        lay["pairs"] += I.size; lay["cancel"] += int((~nz).sum()); lay["wrong"] += int(wrong.sum())
        # one weight bit + one checksum bit (two 32-bit checksum words)
        eps = np.concatenate([1 << np.arange(31, dtype=np.int64), [-(1 << 31)]])
        eps = np.concatenate([eps, -eps])                # either direction, counted as half weight below
        D, E = np.meshgrid(d, eps, indexing="ij"); J = np.repeat(idx, 1)[:, None]
        # C1 hit: s1 = D - E, s2 = J*D ; C2 hit: s1 = D, s2 = J*D - E
        for s1c, s2c in ((D - E, J * D), (D + 0 * E, J * D - E)):
            nz = s1c != 0
            okc = np.zeros_like(nz); okc[nz] = (s2c[nz] % s1c[nz] == 0)
            lc = np.zeros_like(s1c); lc[okc] = s2c[okc] // s1c[okc]
            w = okc & (lc >= 1) & (lc <= G) & (lc != J)
            lay["ck_pairs"] += D.size // 2; lay["ck_wrong"] += int(w.sum()) / 2
    rows.append((n, G, lay))
    for k in lay: tot[k] = tot.get(k, 0) + lay[k]
for n, G, l in rows:
    print(f"{n[-28:]:<28} G={G:4d}  double faults: cancel {l['cancel']/l['pairs']:.3f} wrong repair {l['wrong']/l['pairs']:.3f} | weight+checksum: wrong repair {l['ck_wrong']/l['ck_pairs']:.3f}")
print("all", {k: round(tot[k]/tot['pairs'], 3) for k in ('cancel', 'wrong')}, "ck", round(tot['ck_wrong']/tot['ck_pairs'], 3))
