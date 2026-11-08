# Dictionary-based block-encoding of block-sparse matrices

Numerical experiments for the paper **“Dictionary-based block-encoding for
block-sparse matrices”**.

The repository contains four self-contained Qiskit scripts. Each one builds the
complete block-encoding circuit of one scheme for one partial differential
equation and then checks, by state-vector simulation, that the encoded operator
is the Schrödingerized Hamiltonian divided by its subnormalization:

```
<0| U |0>  =  Ĥ / α
```

The two schemes of the paper crossed with the two equations give the four
scripts:

| script | scheme | equation |
|---|---|---|
| `code1_dictionary_wave.py` | Sec. 2 — keys = **subnormalization** | acoustic wave equation |
| `code2_dictionary_tidal.py` | Sec. 2 — keys = **subnormalization** | Laplace tidal equation |
| `code3_blockclass_wave.py` | Sec. 4 — keys = **the blocks themselves** | acoustic wave equation |
| `code4_blockclass_tidal.py` | Sec. 4 — keys = **the blocks themselves** | Laplace tidal equation |

---

## 1. The two schemes

**Sec. 2 — classification by subnormalization (Theorem 1).**
The non-zero blocks of the block matrix are stored in a dictionary whose keys
collect blocks that share the same subnormalization factor β_ij = A_l.  The
circuit is

```
PREP → O_c → U → SWAP(row, col) → O_r → UNPREP
```

`O_c` writes the block row c_l(j) into a working register and `O_r` clears it
again.  The subnormalization of the encoding is α = Σ_l A_l, and the controlled
encoding `U` is controlled on the 2t block-position bits.

**Sec. 4 — classification by the blocks themselves (Theorem 6).**
A key now collects blocks that are *equal as matrices*, so a single encoding U_l
serves every block of the key.  As a consequence `U` is controlled on the key
register alone — m = ⌈log₂ s₀⌉ qubits instead of 2t bits — and both `O_r` and the
intermediate SWAP disappear.  When every key is a constant shift, c_l(j) = j + d_l,
the index oracle can be realised *in place*: `O_c` overwrites the column register
with a controlled increment.  That is the layout used in `code3`.

The trade-off, visible in the results below, is that the finer keys of the
second scheme charge more distinct blocks, hence a **larger** subnormalization
α, in exchange for a **shorter** controlled encoding.

---

## 2. The two model problems

### 2.1 Acoustic wave equation (Sec. 5.1)

```
d²p/dx²  −  (1/c²) d²p/dt²  =  0 ,      x ∈ [0, Lx] ,  Lx = 1 ,  c = 1
```

* `N_x` equally spaced nodes, `h_x = L_x / (N_x − 1)`; three-point central
  differences in the interior, and the two boundary rows of the second-difference
  matrix are replaced by one-sided differences `[−1, 1] / h_x²`.
* First-order form with `u = (p, ∂_t p)`,

```
A = [ 0    I ]        H1 = (A + A†)/2 ,   H2 = i (A − A†)/2 ,
    [ L_d  0 ]
```

  where `L_d` is the discrete second-derivative matrix.
* Spectral discretization in ξ with `N_ξ` modes,

```
μ_k = k − N_ξ/2 ,   k = 0, …, N_ξ − 1 ,      D_μ = diag(μ_0, …, μ_{N_ξ−1})
```

* Partitioning Ĥ = H1 ⊗ D_μ − H2 ⊗ I according to the dimension of H1 and H2
  gives a **2N_x × 2N_x grid of blocks**, each block `N_ξ × N_ξ` and diagonal:

```
Ĥ_ij = (H1)_ij D_μ − (H2)_ij I
```

`code1` and `code3` build exactly this structure.  The simulated cases are
`N_x = 4, N_ξ = 8` (a 64 × 64 operator) and `N_x = 8, N_ξ = 16` (256 × 256).

### 2.2 Laplace tidal equation (Sec. 5.2)

The linearized shallow-water system on a rotating sphere:

```
∂_t u − 2Ω sinθ v + (1/(a cosθ)) ∂_λ η        = 0
∂_t v + 2Ω sinθ u + (1/a) ∂_θ η               = 0
∂_t η + (gH/(a cosθ)) (∂_λ u + ∂_θ(cosθ v))   = 0
```

Here `u, v` are the zonal and meridional velocities, `η` the geopotential height
of the free surface, `Ω` the angular velocity, `a` the radius, `H` the equivalent
depth, `λ ∈ [0, 2π)` the longitude and `θ ∈ (−π/2, π/2)` the latitude; the
Coriolis parameter is `f = 2Ω sinθ`, and the coefficients `1/cosθ` are singular
at the poles.

Discretization used in `code2` and `code4` (function `build_tidal`):

* **Latitude** — staggered grid

```
θ_n = −π/2 + (n + 1/2) h_θ ,   h_θ = π/N_θ ,   n = 0, …, N_θ − 1
```

  staggered by half a mesh so that no node sits at the poles; central differences
  in the interior and one-sided differences in the two rows next to the poles.
  The divergence term `∂_θ(cosθ v)` is discretized by the same stencil applied to
  the product `cosθ v`.
* **Longitude** — `N_λ` equally spaced nodes, `h_λ = 2π/N_λ`, periodic, so the
  central-difference operator is circulant and diagonal in the zonal wavenumber
  basis with symbol `i sin(2π k / N_λ) / h_λ`.  The code therefore reduces to **a
  single zonal wavenumber** `k`; this is what makes every block diagonal.
  Keeping the full two-dimensional operator instead would multiply the number of
  blocks by `N_λ`.
* **Ordering** — unknowns ordered by field and then by latitude node,
  `U = (u, v, η)ᵀ`, which gives a **3N_θ × 3N_θ grid of blocks**, each block
  `N_ξ × N_ξ` and diagonal, with the 3 × 3 block structure

```
      [       0             F             −(1/a) S D_λ   ]
A_N = [      −F            0             −(1/a) D_θ     ]
      [ −(gH/a) S D_λ   −(gH/a) S D_θ C        0        ]

F = diag(f_n) ,   S = diag(secθ_n) ,   C = diag(cosθ_n)
```

  where `D_λ` is the circulant and `D_θ` the tridiagonal matrix.
* **Padding** — `3N_θ` is not a power of two, so the block positions are padded
  to `2^t` while each block keeps `2^r = N_ξ` entries, matching the register
  widths of the circuit.

The propagation in ξ is the same as for the wave equation (warped phase
transformation followed by Fourier spectral discretization), so the Hamiltonian
has the same form `Ĥ = H1 ⊗ D_μ − H2 ⊗ I` and the same diagonal-block encoding
applies.

`code2` and `code4` run `N_θ = 4, N_λ = 16, N_ξ = 4` (a 64 × 64 operator, fully
simulated) and `N_θ = 8, N_λ = 16, N_ξ = 8` (256 × 256, structure only: at 20–22
qubits the full state-vector extraction is prohibitively slow).

### 2.3 Why both problems end up in the same encoding problem

In both cases every non-zero block is a diagonal matrix

```
B_ij = diag(b_ij,0 , …, b_ij,2^r−1)
```

and is encoded at its spectral norm

```
β_ij = ||B_ij|| = max_k |b_ij,k|
```

with one ancilla qubit and a multiplexed single-qubit rotation (Sec. 3.4 of the
paper).  This is the only place where the physics of the two equations enters the
circuit: everything else is the dictionary, the index oracles and the state
preparation.

---

## 3. Repository layout

```
code1_dictionary_wave.py     Sec. 2 scheme, wave equation      (2 cases)
code2_dictionary_tidal.py    Sec. 2 scheme, tidal equation     (2 cases)
code3_blockclass_wave.py     Sec. 4 scheme, wave equation      (2 cases)
code4_blockclass_tidal.py    Sec. 4 scheme, tidal equation     (2 cases)
results_code1.json           machine-readable output of code1
results_code2.json           …
results_code3.json           …
results_code4.json           …
results_summary.csv          one line per case, all four scripts
README.md                    this file
requirements.txt             Python dependencies
```

Every script is self-contained: it imports none of the others and needs no data
files.  Each one prints the circuit structure, the circuit width, the
subnormalization and the encoding error, and writes its results to
`results_codeN.json` in the working directory.

---

## 4. Circuits and registers

`m = ⌈log₂ s₀⌉` is the width of the key register, `t` the width of the block
index, `r` the width of the intra-block index and `a = 1` the ancilla of the
diagonal-block encoding.

Two-register layout (both Sec. 2 scripts, and both tidal scripts):

| register | width | meaning |
|---|---|---|
| `idx`  | m | dictionary key |
| `del0` | 1 | validity flag written by `O_c` |
| `del1` | 1 | validity flag written by `O_r` |
| `row`  | t | working register, `O_c` writes c_l(j) |
| `col`  | t | block column j on input, block row i on output |
| `anc`  | 1 | ancilla of the diagonal-block encoding |
| `sys`  | r | intra-block index k |

In-place layout (Sec. 4 scheme, wave equation):

| register | width | meaning |
|---|---|---|
| `idx`  | m | dictionary key |
| `del0` | 1 | validity flag written by `O_c` |
| `col`  | t | block column, overwritten in place by the block row |
| `anc`  | 1 | ancilla of the diagonal-block encoding |
| `sys`  | r | intra-block index k |

The multiplexed rotation `U = Σ_c |c><c| ⊗ R_c` is implemented with `UCRY` and
`UCRZ` gates; the index oracles are built from multi-controlled NOT gates whose
controls are drawn from the key and block-position registers.

---

## 5. Results

The numbers below are produced by the scripts as they are committed here and
coincide with the table of the paper.  `α` is the subnormalization, `s` the
number of non-zero blocks, `s₀` the number of dictionary keys, and the error is
the largest absolute deviation between the simulated `<0|U|0>` and `Ĥ/α`.

| script | case | dim | s | s₀ | α | qubits | max error |
|---|---|---|---|---|---|---|---|
| `code1` | wave N_x=4, N_ξ=8  | 64  | 20 | 4  | 89.17   | 14 | 1.6e-13 |
| `code1` | wave N_x=8, N_ξ=16 | 256 | 44 | 4  | 979.8   | 17 | 4.3e-13 |
| `code2` | tidal N_θ=4, N_λ=16, N_ξ=4 | 64 | 32 | 8  | 1.069e-2 | 16 | 1.2e-12 |
| `code2` | tidal N_θ=8, N_λ=16, N_ξ=8 (structure only) | 256 | 64 | 16 | 5.242e-2 | 20 | — |
| `code3` | wave N_x=4, N_ξ=8  | 64  | 20 | 8  | 178.3   | 11 | 9.2e-14 |
| `code3` | wave N_x=8, N_ξ=16 | 256 | 44 | 8  | 1959.6  | 13 | 1.5e-13 |
| `code4` | tidal N_θ=4, N_λ=16, N_ξ=4 | 64 | 32 | 24 | 2.996e-2 | 18 | 1.1e-12 |
| `code4` | tidal N_θ=8, N_λ=16, N_ξ=8 (structure only) | 256 | 64 | 40 | 1.230e-1 | 22 | — |

For the two cases marked *structure only* the circuits are built and reported but
not simulated, because extracting the full matrix at 20–22 qubits is
prohibitively slow; only the structural data are given, as in the paper.

Three remarks on the comparison:

* **Wave equation.** The Sec. 4 scheme needs three to four qubits fewer (11 and
  13 versus 14 and 17) because `U` is controlled on the key register instead of
  the block positions and the in-place index oracle removes the row register,
  the SWAP and `O_r`.
* **Tidal equation.** Here `code4` keeps the two-register layout — its keys mix
  `+d` and `−d` offsets, so the in-place oracle cannot be used — and its keys
  are finer (s₀ = 24 and 40 versus 8 and 16), so the key register is wider and the
  circuits need two qubits more (18 and 22 versus 16 and 20).  What is saved is
  the width of the controlled encoding, `m` qubits instead of `2t`, and the
  number of distinct block encodings.
* **Subnormalization.** The Sec. 4 scheme has the larger α (178.3 versus 89.17;
  1959.6 versus 979.8): every distinct block is charged once, whereas blocks
  sharing a norm are charged once in the Sec. 2 scheme.  This is exactly the
  trade-off discussed in the paper.

---

## 6. Requirements and running

```
python >= 3.9
qiskit >= 2.0          (developed and tested with 2.5.2)
qiskit-aer             (state-vector simulation)
numpy
```

```bash
pip install -r requirements.txt
```

Run the scripts one at a time:

```bash
python code3_blockclass_wave.py     # smallest circuits: start here
python code1_dictionary_wave.py
python code2_dictionary_tidal.py
python code4_blockclass_tidal.py
```

A run takes roughly one to five minutes on a laptop (single thread); almost all
of the time is spent in the state-vector simulation, not in building the circuit.

---

## 7. Citation

If you use this code, please cite

```
Yuhao Song, Chunlin Yang, Hongmei Yao,
Dictionary-based block-encoding for block-sparse matrices.
```

