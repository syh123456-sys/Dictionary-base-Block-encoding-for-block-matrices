#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 1 -- dictionary-based block encoding, Section 2
(classification by subnormalization factor), applied to the
Schroedingerized 1-D acoustic wave equation.

H^ = H1 (x) D_mu - H2 (x) I is partitioned into 2^t x 2^t blocks
according to the dimension of H1 and H2; every non-zero block
H^_ij = (H1)_ij D_mu - (H2)_ij I is diagonal and is encoded at its
spectral norm beta_ij = ||H^_ij|| with a single ancilla qubit.
U is controlled on the greedily selected block-position bits.

Outputs: subnormalization alpha, number of keys s0, circuit width and
size, and max |<0|U|0> - H^/alpha| for the two test cases
(Nx, Nxi) = (4, 8) and (8, 16)."""

import json
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate, UCRYGate, UCRZGate
from qiskit_aer import AerSimulator

import json
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate, UCRYGate, UCRZGate
from qiskit_aer import AerSimulator


# --------------------------------------------------------------------------- #
# 1.  Physical models
# --------------------------------------------------------------------------- #
def build_wave(Nx, Nxi, Lx=1.0, L=1.0):
    """Schroedingerized 1-D acoustic wave equation, Section 5.1.1."""
    hx = Lx / (Nx - 1)
    inv_h2 = 1.0 / hx ** 2
    D2 = -2.0 * np.eye(Nx) + np.diag(np.ones(Nx - 1), 1) + np.diag(np.ones(Nx - 1), -1)
    D2[0, 0], D2[0, 1] = -1.0, 1.0
    D2[-1, -1], D2[-1, -2] = -1.0, 1.0
    Ld = inv_h2 * D2
    Z = np.zeros((Nx, Nx))
    A = np.block([[Z, np.eye(Nx)], [Ld, Z]])
    Ad = np.block([[Z, Ld], [np.eye(Nx), Z]])
    H1 = 0.5 * (A + Ad)
    H2 = 0.5j * (A - Ad)
    mu = np.arange(Nxi) - Nxi / 2.0
    Hhat = np.kron(H1, np.diag(mu)) - np.kron(H2, np.eye(Nxi))
    return Hhat, H1, H2, mu


def build_tidal(Nth, Nlam, Nxi, a=6.371e6, Omega=7.292e-5, g=9.81, H=1000.0,
                zonal_mode=1):
    """Laplace tidal equation, Section 5.1.2."""
    hlam = 2.0 * np.pi / Nlam
    hth = np.pi / Nth
    th = -np.pi / 2.0 + (np.arange(Nth) + 0.5) * hth
    kappa = np.sin(2.0 * np.pi * zonal_mode / Nlam) / hlam

    Dth = np.zeros((Nth, Nth))
    for n in range(1, Nth - 1):
        Dth[n, n - 1] = -1.0 / (2 * hth)
        Dth[n, n + 1] = 1.0 / (2 * hth)
    Dth[0, 0], Dth[0, 1] = -1.0 / hth, 1.0 / hth
    Dth[-1, -1], Dth[-1, -2] = 1.0 / hth, -1.0 / hth

    Ddiv = np.zeros((Nth, Nth))
    for n in range(1, Nth - 1):
        Ddiv[n, n - 1] = -np.cos(th[n - 1]) / (2 * hth)
        Ddiv[n, n + 1] = np.cos(th[n + 1]) / (2 * hth)
    Ddiv[0, 0], Ddiv[0, 1] = -np.cos(th[0]) / hth, np.cos(th[1]) / hth
    Ddiv[-1, -1], Ddiv[-1, -2] = np.cos(th[-1]) / hth, -np.cos(th[-2]) / hth

    sec = 1.0 / np.cos(th)
    f = 2.0 * Omega * np.sin(th)
    Z = np.zeros((Nth, Nth))
    A = np.block([
        [Z, np.diag(f), -1j * kappa / a * np.diag(sec)],
        [-np.diag(f), Z, -(1.0 / a) * Dth],
        [-1j * kappa * g * H / a * np.diag(sec), -(g * H / a) * Ddiv, Z],
    ])
    H1 = 0.5 * (A + A.conj().T)
    H2 = 0.5j * (A - A.conj().T)
    n = H1.shape[0]
    t = int(np.ceil(np.log2(n)))
    if 2 ** t != n:
        P = np.zeros((2 ** t, 2 ** t), dtype=complex)
        P[:n, :n] = H1
        H1 = P
        P = np.zeros((2 ** t, 2 ** t), dtype=complex)
        P[:n, :n] = H2
        H2 = P
    mu = np.arange(Nxi) - Nxi / 2.0
    Hhat = np.kron(H1, np.diag(mu)) - np.kron(H2, np.eye(Nxi))
    return Hhat, H1, H2, mu


# --------------------------------------------------------------------------- #
# 2.  Dictionary
# --------------------------------------------------------------------------- #
def nonzero_blocks(H1, H2, mu, tol=1e-14):
    M = H1.shape[0]
    out = []
    for i in range(M):
        for j in range(M):
            h1, h2 = H1[i, j], H2[i, j]
            if abs(h1) <= tol and abs(h2) <= tol:
                continue
            d = h1 * mu - h2
            beta = float(np.max(np.abs(d)))
            if beta < 1e-15:
                continue
            out.append((i, j, d.copy(), beta, complex(h1), complex(h2)))
    return out


def _chain_order(block, M):
    i, j = block[0], block[1]
    half = M // 2
    if i < half <= j:
        return (0, i - (j - half), i)
    if j < half <= i:
        return (1, (i - half) - j, i)
    return (2, i - j, i)


def build_keys(blocks, algo, M, decimals=10, split_offsets=False):
    """Dictionary keys.  split_offsets splits equal-block classes by the
    offset i - j, so that every key is a constant shift c_l(j) = j + d_l."""
    groups = {}
    for b in blocks:
        if algo == "sec2":
            cls = round(b[3], decimals)
        else:
            cls = (round(b[4].real, decimals), round(b[4].imag, decimals),
                   round(b[5].real, decimals), round(b[5].imag, decimals))
            if split_offsets:
                cls = cls + (b[0] - b[1],)
        groups.setdefault(cls, []).append(b)

    keys = []
    for cls in sorted(groups, key=lambda x: (isinstance(x, tuple), x)):
        bl = sorted(groups[cls], key=lambda b: _chain_order(b, M))
        matchings = []
        for b in bl:
            i, j = b[0], b[1]
            for mt in matchings:
                if i not in mt["rows"] and j not in mt["cols"]:
                    mt["rows"].add(i)
                    mt["cols"].add(j)
                    mt["blocks"].append(b)
                    break
            else:
                matchings.append({"rows": {i}, "cols": {j}, "blocks": [b]})
        for mt in matchings:
            blk = mt["blocks"]
            keys.append({
                "A": float(blk[0][3]),
                "blocks": blk,
                "Sc": set(x[1] for x in blk),
                "c": {x[1]: x[0] for x in blk},
                "Sr": set(x[0] for x in blk),
                "r": {x[0]: x[1] for x in blk},
            })
    return keys


def is_shift(key):
    return len({key["c"][j] - j for j in key["c"]}) == 1


def greedy_positions(blocks, t):
    strings = [[(b[0] >> q) & 1 for q in range(t)] + [(b[1] >> q) & 1 for q in range(t)]
               for b in blocks]
    groups = [list(range(len(strings)))]
    selected = []
    while not all(len(g) == 1 for g in groups):
        best, best_score = None, -1
        for pos in range(2 * t):
            if pos in selected:
                continue
            score = 0
            for g in groups:
                n0 = sum(1 for x in g if strings[x][pos] == 0)
                score += n0 * (len(g) - n0)
            if score > best_score:
                best_score, best = score, pos
        if best is None:
            break
        selected.append(best)
        ng = []
        for g in groups:
            g0 = [x for x in g if strings[x][best] == 0]
            g1 = [x for x in g if strings[x][best] == 1]
            ng += [g0] if g0 else []
            ng += [g1] if g1 else []
        groups = ng
    return selected


# --------------------------------------------------------------------------- #
# 3.  Gates
# --------------------------------------------------------------------------- #
def _mcx(qc, controls, target, ones):
    cs = "".join(str(int(o)) for o in reversed(ones))
    qc.mcx(list(controls), target, ctrl_state=cs)


def _cry(qc, theta, controls, target):
    if abs(theta) < 1e-15:
        return
    qc.append(RYGate(theta).control(len(controls)), list(controls) + [target])


def _add_const(qc, bits, delta, ctrls, vals):
    """bits <- bits + delta (mod 2^len(bits)), controlled on ctrls == vals."""
    step = 1 if delta >= 0 else -1
    for _ in range(abs(delta)):
        for q in reversed(range(len(bits))):
            c = list(bits[:q]) + list(ctrls)
            v = ([1] * q if step > 0 else [0] * q) + list(vals)
            _mcx(qc, c, bits[q], v)


def _prep_records(amps, m):
    state = np.array(amps, dtype=float).copy()
    recs = []
    for layer in range(m):
        step = 2 ** (m - layer - 1)
        for start in range(0, 2 ** m, 2 * step):
            left = float(np.sqrt(np.sum(state[start:start + step] ** 2)))
            right = float(np.sqrt(np.sum(state[start + step:start + 2 * step] ** 2)))
            total = float(np.hypot(left, right))
            if total < 1e-15:
                continue
            recs.append((layer, start // (2 * step),
                         2.0 * np.arccos(np.clip(left / total, -1.0, 1.0))))
            if left > 1e-15:
                state[start:start + step] *= total / left
            if right > 1e-15:
                state[start + step:start + 2 * step] *= total / right
    return recs


def build_prep(qc, amps, m, offset):
    for layer, bidx, theta in _prep_records(amps, m):
        target = offset + (m - 1 - layer)
        if layer == 0:
            qc.ry(theta, target)
        else:
            ctrls = [offset + (m - 1 - p) for p in range(layer)]
            flips = [ctrls[p] for p in range(layer) if not ((bidx >> (layer - 1 - p)) & 1)]
            for q in flips:
                qc.x(q)
            _cry(qc, theta, ctrls, target)
            for q in flips:
                qc.x(q)


def build_unprep(qc, amps, m, offset):
    recs = _prep_records(amps, m)
    for layer in reversed(range(m)):
        for lay, bidx, theta in sorted([r for r in recs if r[0] == layer],
                                       key=lambda r: r[1], reverse=True):
            target = offset + (m - 1 - lay)
            if lay == 0:
                qc.ry(-theta, target)
            else:
                ctrls = [offset + (m - 1 - p) for p in range(lay)]
                flips = [ctrls[p] for p in range(lay) if not ((bidx >> (lay - 1 - p)) & 1)]
                for q in flips:
                    qc.x(q)
                _cry(qc, -theta, ctrls, target)
                for q in flips:
                    qc.x(q)


def index_oracle(qc, keys, m, t, del_q, row_off, addr_off, addr_set, addr_map):
    """Three-stage sparse-Boolean-memory index oracle (two-register layout)."""
    qc.x(del_q)
    for l, key in enumerate(keys):
        for a in key[addr_set]:
            ctrls = [b for b in range(m)] + [addr_off + b for b in range(t)]
            ones = [((l >> b) & 1) for b in range(m)] + [((a >> b) & 1) for b in range(t)]
            _mcx(qc, ctrls, del_q, ones)
    for l, key in enumerate(keys):
        for a in key[addr_set]:
            out = key[addr_map][a]
            ctrls = [del_q] + [b for b in range(m)] + [addr_off + b for b in range(t)]
            ones = [0] + [((l >> b) & 1) for b in range(m)] + [((a >> b) & 1) for b in range(t)]
            for b in range(t):
                if (out >> b) & 1:
                    _mcx(qc, ctrls, row_off + b, ones)


def index_oracle_inplace(qc, keys, m, t, del_q, col_off):
    """In-place index oracle of Theorem 6: col <- c_l(col) = col + d_l."""
    qc.x(del_q)
    for l, key in enumerate(keys):
        for a in key["Sc"]:
            ctrls = [b for b in range(m)] + [col_off + b for b in range(t)]
            ones = [((l >> b) & 1) for b in range(m)] + [((a >> b) & 1) for b in range(t)]
            _mcx(qc, ctrls, del_q, ones)
    for l, key in enumerate(keys):
        j0 = next(iter(key["c"]))
        delta = key["c"][j0] - j0
        if delta == 0:
            continue
        ctrls = [b for b in range(m)] + [del_q]
        vals = [((l >> b) & 1) for b in range(m)] + [0]
        _add_const(qc, [col_off + b for b in range(t)], delta, ctrls, vals)


# --------------------------------------------------------------------------- #
# 4.  Full block-encoding circuit
# --------------------------------------------------------------------------- #
def build_circuit(H1, H2, mu, algo, use_greedy=True, layout="auto",
                  split_offsets=False):
    """layout : 'auto' | 'inplace' | 'tworeg'

    The dictionary follows the paper: blocks are first grouped by the
    subnormalization factor (Sec. 2) or by the block itself (Sec. 4), and every
    class is then split by a first-fit matching of block rows and columns.
    If the resulting keys are constant shifts c_l(j) = j + d_l, the index
    oracle is realised in place (Theorem 6: m + 1 + a ancillas); otherwise the
    row register and the O_r uncompute of Theorem 1 are kept.
    split_offsets=True additionally splits a class by the offset i - j, which
    makes every key a shift at the price of more keys (and a larger alpha).
    """
    blocks = nonzero_blocks(H1, H2, mu)
    M = H1.shape[0]
    t = int(round(np.log2(M)))
    r = int(round(np.log2(len(mu))))

    keys = build_keys(blocks, algo, M, split_offsets=(split_offsets and algo == "sec4"))
    inplace = (algo == "sec4" and layout != "tworeg" and all(is_shift(k) for k in keys))
    if layout == "inplace" and not inplace:
        raise RuntimeError("layout='inplace' requires constant-shift keys")

    s0 = len(keys)
    m = max(1, int(np.ceil(np.log2(max(s0, 2)))))
    alpha = float(sum(k["A"] for k in keys))

    if inplace:
        del0 = m
        col_off = m + 1
        anc = m + 1 + t
        sys_off = m + 2 + t
        nq = m + 2 + t + r
        off = dict(idx=0, del0=del0, col=col_off, anc=anc, sys=sys_off)
    else:
        del0, del1 = m, m + 1
        row_off = m + 2
        col_off = m + 2 + t
        anc = m + 2 + 2 * t
        sys_off = m + 3 + 2 * t
        nq = m + 3 + 2 * t + r
        off = dict(idx=0, del0=del0, del1=del1, row=row_off, col=col_off,
                   anc=anc, sys=sys_off)

    qc = QuantumCircuit(nq)
    amps = np.zeros(2 ** m)
    for l, k in enumerate(keys):
        amps[l] = np.sqrt(k["A"] / alpha)
    build_prep(qc, amps, m, 0)

    if inplace:
        index_oracle_inplace(qc, keys, m, t, del0, col_off)
    else:
        index_oracle(qc, keys, m, t, del0, off["row"], col_off, "Sc", "c")

    if algo == "sec2":
        pos = greedy_positions(blocks, t) if use_greedy else list(range(2 * t))
        phi = np.zeros((2 ** len(pos), 2 ** r))
        th = np.zeros((2 ** len(pos), 2 ** r))
        for (i, j, d, beta, h1, h2) in blocks:
            proj = 0
            for w, p in enumerate(pos):
                if p < t:
                    proj |= ((i >> p) & 1) << w
                else:
                    proj |= ((j >> (p - t)) & 1) << w
            phi[proj, :] = 2.0 * np.arccos(np.clip(np.abs(d) / beta, 0.0, 1.0))
            th[proj, :] = np.angle(d)
        ctrl = [(off["row"] + p if p < t else col_off + p - t) for p in pos] + \
               [sys_off + b for b in range(r)]
    else:
        phi = np.zeros((2 ** m, 2 ** r))
        th = np.zeros((2 ** m, 2 ** r))
        for l, key in enumerate(keys):
            d = key["blocks"][0][2]
            phi[l, :] = 2.0 * np.arccos(np.clip(np.abs(d) / key["A"], 0.0, 1.0))
            th[l, :] = np.angle(d)
        ctrl = [b for b in range(m)] + [sys_off + b for b in range(r)]

    qc.append(UCRYGate(list(phi.T.reshape(-1))), [anc] + ctrl)
    qc.append(UCRZGate(list(-2.0 * th.T.reshape(-1))), [anc] + ctrl)

    if not inplace:
        for b in range(t):
            qc.swap(off["row"] + b, col_off + b)
        index_oracle(qc, keys, m, t, del1, off["row"], col_off, "Sr", "r")

    build_unprep(qc, amps, m, 0)

    meta = dict(algo=algo, m=m, t=t, r=r, s=len(blocks), s0=s0, alpha=alpha,
                keys=keys, n_qubits=nq, layout=("inplace" if inplace else "tworeg"),
                n_ctrl=(len(greedy_positions(blocks, t)) if algo == "sec2" else m),
                n_ctrl_max=2 * t, offsets=off)
    return qc, meta


# --------------------------------------------------------------------------- #
# 5.  Verification
# --------------------------------------------------------------------------- #
def encoded_columns(qc, meta, cols=None, verbose=True):
    m, t, r = meta["m"], meta["t"], meta["r"]
    off = meta["offsets"]
    n_sys = t + r
    if cols is None:
        cols = list(range(2 ** n_sys))
    qt = qc.copy()
    for _ in range(12):
        if set(g.operation.name for g in qt.data) <= {"u", "cx", "x", "rz", "ry", "h", "p"}:
            break
        qt = qt.decompose()
    qt.save_statevector()
    sim = AerSimulator(method="statevector")
    M = np.zeros((2 ** n_sys, len(cols)), complex)
    t0 = time.time()
    for c, j in enumerate(cols):
        circ = QuantumCircuit(qt.num_qubits)
        for q in range(t):
            if (j >> q) & 1:
                circ.x(off["col"] + q)
        for q in range(r):
            if (j >> (t + q)) & 1:
                circ.x(off["sys"] + q)
        circ.compose(qt, inplace=True)
        sv = np.asarray(sim.run(circ, shots=1).result().get_statevector())
        jc, kk = j % 2 ** t, j // 2 ** t
        base = kk << off["sys"]
        for i in range(2 ** t):
            M[i + kk * 2 ** t, c] = sv[base | (i << off["col"])]
        if verbose and (c + 1) % 8 == 0:
            print(f"      {c+1}/{len(cols)} columns ({time.time()-t0:.1f} s)", flush=True)
    return M, cols


def perm_std_to_reg(t, r):
    perm = np.zeros(2 ** (t + r), dtype=np.int64)
    for i in range(2 ** t):
        for k in range(2 ** r):
            perm[i + k * 2 ** t] = i * 2 ** r + k
    return perm


def verify(qc, meta, Hhat, cols=None, verbose=True):
    t, r, alpha = meta["t"], meta["r"], meta["alpha"]
    M, cols = encoded_columns(qc, meta, cols, verbose)
    perm = perm_std_to_reg(t, r)
    ref = (Hhat[np.ix_(perm, perm)] / alpha)[:, cols]
    return dict(max_abs=float(np.abs(M - ref).max()),
                max_rel=float(np.abs(M - ref).max() / max(np.abs(ref).max(), 1e-30)),
                n_cols=len(cols), n_sys=t + r)


# --------------------------------------------------------------------------- #
# 6.  Reporting
# --------------------------------------------------------------------------- #
def report(name, Hhat, qc, meta, res, extra=None):
    t, r, alpha = meta["t"], meta["r"], meta["alpha"]
    print("-" * 78)
    print(f"  {name}")
    print("-" * 78)
    print(f"  operator dimension         : {Hhat.shape[0]} x {Hhat.shape[1]}")
    print(f"  block structure            : {2**t} x {2**t} blocks of {2**r} x {2**r}")
    print(f"  non-zero blocks       s    : {meta['s']}")
    print(f"  dictionary keys       s0   : {meta['s0']}   (m = {meta['m']} key qubits)")
    print(f"  subnormalization  alpha    : {alpha:.6e}")
    print(f"  ||H||_2                    : {np.linalg.norm(Hhat, 2):.6e}")
    print(f"  sum |H_mn|                 : {np.sum(np.abs(Hhat)):.6e}")
    print(f"  alpha / sum|H_mn|          : {alpha/np.sum(np.abs(Hhat)):.4f}")
    print(f"  layout of O_c              : {meta['layout']}")
    print(f"  block-position controls    : {meta['n_ctrl']} of {meta['n_ctrl_max']} used by U")
    print(f"  circuit qubits             : {meta['n_qubits']}")
    print(f"  circuit size / depth       : {qc.size()} / {qc.depth()}")
    print(f"  max |M - H/alpha|          : {res['max_abs']:.3e}"
          f"   (relative {res['max_rel']:.3e})")
    print(f"  columns verified           : {res['n_cols']} / {2**res['n_sys']}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:<27}: {v}")
    print()


def resource_estimate(H1, H2, mu, algo):
    blocks = nonzero_blocks(H1, H2, mu)
    M = H1.shape[0]
    t = int(round(np.log2(M)))
    r = int(round(np.log2(len(mu))))
    keys = build_keys(blocks, algo, M, split_offsets=False)
    inplace = (algo == "sec4" and all(is_shift(k) for k in keys))
    s0 = len(keys)
    m = max(1, int(np.ceil(np.log2(max(s0, 2)))))
    alpha = float(sum(k["A"] for k in keys))
    nq = (m + 2 + t + r) if inplace else (m + 3 + 2 * t + r)
    return dict(s=len(blocks), s0=s0, m=m, t=t, r=r, alpha=alpha, n_qubits=nq,
                layout=("inplace" if inplace else "tworeg"),
                n_ctrl=(len(greedy_positions(blocks, t)) if algo == "sec2" else m),
                n_ctrl_max=2 * t)


# --------------------------------------------------------------------------- #
#  Main program
# --------------------------------------------------------------------------- #
def run_case(name, model, args, algo, cols, verbose=True):
    t0 = time.time()
    Hhat, H1, H2, mu = model(*args)
    qc, meta = build_circuit(H1, H2, mu, algo=algo)
    res = verify(qc, meta, Hhat, cols=cols, verbose=verbose)
    report(name, Hhat, qc, meta, res, {"wall time": "%.1f s" % (time.time() - t0)})
    return dict(case=name, s=meta["s"], s0=meta["s0"], alpha=meta["alpha"],
                qubits=meta["n_qubits"], layout=meta["layout"],
                ctrl=meta["n_ctrl"], ctrl_max=meta["n_ctrl_max"],
                size=qc.size(), depth=qc.depth(), err=res["max_abs"],
                rel=res["max_rel"], cols=res["n_cols"], dim=Hhat.shape[0])


def summary(rows, fname):
    print("=" * 96)
    print("  summary")
    print("=" * 96)
    print("  %-42s %6s %4s %4s %12s %7s %8s %10s" %
          ("case", "dim", "s", "s0", "alpha", "qubits", "controls", "error"))
    for r in rows:
        print("  %-42s %6d %4d %4d %12.4e %7d %8s %10.2e" %
              (r["case"], r["dim"], r["s"], r["s0"], r["alpha"], r["qubits"],
               "%d/%d" % (r["ctrl"], r["ctrl_max"]), r["err"]))
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\n  results written to", fname)


if __name__ == "__main__":
    print(__doc__)
    rows = []
    rows.append(run_case("Sec.2 dictionary | wave Nx=4  Nxi=8 ", build_wave, (4, 8),
                         "sec2", None))
    rows.append(run_case("Sec.2 dictionary | wave Nx=8  Nxi=16", build_wave, (8, 16),
                         "sec2", list(range(0, 256, 16))))
    summary(rows, "results_code1.json")
