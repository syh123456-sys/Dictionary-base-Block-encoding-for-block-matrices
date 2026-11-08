# Dictionary-based block-encoding of block-sparse matrices

Numerical experiments for the paper **“Dictionary-based block-encoding for
block-sparse matrices”**.  The repository contains four self-contained Qiskit
scripts that build the full block-encoding circuits of the two schemes proposed
in the paper and verify, by state-vector simulation, that

$$\langle 0|\,U\,|0\rangle \;=\; \hat H/\alpha ,$$

where $\hat H$ is the Schrödingerized Hamiltonian of a partial differential
equation and $\alpha$ is the subnormalization of the encoding.

The two schemes of the paper are crossed with the two equations, which gives the
four scripts:

| script | scheme | equation |
|---|---|---|
| `code1_dictionary_wave.py` | Sec. 2 — classification by **subnormalization** | acoustic wave equation |
| `code2_dictionary_tidal.py` | Sec. 2 — classification by **subnormalization** | Laplace tidal equation |
| `code3_blockclass_wave.py` | Sec. 4 — classification by **the blocks themselves** | acoustic wave equation |
| `code4_blockclass_tidal.py` | Sec. 4 — classification by **the blocks themselves** | Laplace tidal equation |

---

## 1. The two schemes in one paragraph each

**Classification by subnormalization (Sec. 2, Theorem 1).**  The non-zero blocks
of $B=\sum_{i,j}|i\rangle\langle j|\otimes B_{ij}$ are recorded in a dictionary
whose keys collect blocks with the same subnormalization factor
$\beta_{ij}=A_l$.  The encoding is assembled from two state-preparation oracles
and two index oracles,

```
PREP → O_c → U → SWAP(row,col) → O_r → UNPREP ,
```

where $O_c$ writes the block row $c_l(j)$ into a working register and $O_r$
clears it again.  The subnormalization is $\alpha=\sum_l A_l$.

**Classification by the blocks themselves (Sec. 4, Theorem 6).**  A key now
collects blocks that are *equal as matrices*, so a single encoding $U_l$ serves
every block of the key.  Consequently $U$ is controlled on the key register
alone ($m=\lceil\log_2 s_0\rceil$ qubits instead of the $2t$ block-position
bits), and both $O_r$ and the intermediate SWAP disappear.  When every key is a
constant shift, $c_l(j)=j+d_l$, the index oracle can be realised *in place*
($O_c$ overwrites the column register with a controlled increment), which is the
layout used in `code3`.

In both schemes the blocks of $\hat H=H_1\otimes D_\mu-H_2\otimes I$ are
diagonal,

$$\hat H_{ij}=(H_1)_{ij}D_\mu-(H_2)_{ij}I ,$$

and are encoded at their spectral norm $\beta_{ij}=\|\hat H_{ij}\|$ with a single
ancilla qubit and a multiplexed single-qubit rotation (Sec. 3.4 of the paper).

---

## 2. Repository layout

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

Every script is self-contained: it does not import the others and needs no data
files.  Each one prints the circuit structure, the circuit width, the
subnormalization and the encoding error, and writes its results to
`results_codeN.json` in the working directory.

---

## 3. Requirements

```
python >= 3.9
qiskit >= 2.0          (developed and tested with 2.5.2)
qiskit-aer             (state-vector simulation)
numpy
```

```bash
pip install -r requirements.txt
```

## 4. Running

```bash
python code3_blockclass_wave.py     # smallest circuits: start here
python code1_dictionary_wave.py
python code2_dictionary_tidal.py
python code4_blockclass_tidal.py
```

A run takes roughly one to five minutes on a laptop (single thread); most of the
time is spent in the state-vector simulation, not in building the circuit.

To reproduce the whole table below in one go:

```bash
for f in code1_dictionary_wave.py code2_dictionary_tidal.py \
         code3_blockclass_wave.py code4_blockclass_tidal.py; do python $f; done
```

---

## 5. Results

The numbers below are produced by the scripts as they are committed here and
coincide with the table of the paper.  `α` is the subnormalization, `s` the
number of non-zero blocks, `s₀` the number of dictionary keys, and the error is
the largest absolute deviation between the simulated $\langle 0|U|0\rangle$ and
$\hat H/\alpha$.

| script | case | dim | $s$ | $s_0$ | $\alpha$ | qubits | max error |
|---|---|---|---|---|---|---|---|
| `code1` | wave $N_x=4,\ N_\xi=8$  | 64  | 20 | 4  | 89.17   | 14 | $1.6\times10^{-13}$ |
| `code1` | wave $N_x=8,\ N_\xi=16$ | 256 | 44 | 4  | 979.8   | 17 | $4.3\times10^{-13}$ |
| `code2` | tidal $N_\theta=4,\ N_\xi=4$ | 64 | 32 | 8  | $1.069\times10^{-2}$ | 16 | $1.2\times10^{-12}$ |
| `code2` | tidal $N_\theta=8,\ N_\xi=8$ (structure only) | 256 | 64 | 16 | $5.242\times10^{-2}$ | 20 | — |
| `code3` | wave $N_x=4,\ N_\xi=8$  | 64  | 20 | 8  | 178.3   | 11 | $9.2\times10^{-14}$ |
| `code3` | wave $N_x=8,\ N_\xi=16$ | 256 | 44 | 8  | 1959.6  | 13 | $1.5\times10^{-13}$ |
| `code4` | tidal $N_\theta=4,\ N_\xi=4$ | 64 | 32 | 24 | $2.996\times10^{-2}$ | 18 | $1.1\times10^{-12}$ |
| `code4` | tidal $N_\theta=8,\ N_\xi=8$ (structure only) | 256 | 64 | 40 | $1.230\times10^{-1}$ | 22 | — |

For the two tidal cases marked *structure only* the circuits are built and
reported but not simulated: extracting the full matrix at 20–22 qubits is
prohibitively slow, so only the structural data are given (as in the paper).

Two points are worth noting.

* The Sec. 4 scheme (10–13 qubits on the wave equation) needs several qubits
  fewer than the Sec. 2 scheme (14–17), because $U$ is controlled on the key
  register instead of the block positions.
* The subnormalization of the Sec. 4 scheme is **larger** (178.3 vs 89.17;
  1959.6 vs 979.8), because its keys are finer: every distinct block is charged
  once, whereas blocks sharing a norm are charged once in the Sec. 2 scheme.
  This is the trade-off discussed in the paper.

---

## 6. What the scripts do

### 6.1 Models

**Acoustic wave equation (Sec. 5.1).**

$$\frac{\partial^2 p}{\partial x^2}-\frac{1}{c^2}\frac{\partial^2 p}{\partial t^2}=0 ,
\qquad x\in[0,L_x],\quad L_x=1,\quad c=1 .$$

$N_x$ equally spaced nodes, $h_x=L_x/(N_x-1)$, three-point central differences in
the interior and one-sided differences in the two boundary rows; first-order
form $u=(p,\partial_t p)^T$ with $A=\begin{pmatrix}0&I\\ L_d&0\end{pmatrix}$,
$H_1=(A+A^\dagger)/2$, $H_2=\mathrm{i}(A-A^\dagger)/2$; spectral discretization
in $\xi$ with $\mu_k=k-N_\xi/2$.

**Laplace tidal equation (Sec. 5.2).**  Staggered grid in latitude,
$\theta_n=-\pi/2+(n+\tfrac12)h_\theta$, central differences with one-sided rows
next to the poles, and a single zonal wavenumber in longitude (the circulant
$D_\lambda$ has symbol $\mathrm{i}\sin(2\pi k/N_\lambda)/h_\lambda$, so each
block is diagonal and the block grid is $3N_\theta\times3N_\theta$).  The $3N_\theta$
operator positions are zero-padded to $2^t$, which is what the encoding needs.

### 6.2 Circuits

Both layouts used in the scripts are listed below; `m = ⌈log₂ s₀⌉`,
`t` = block-index width, `r` = width of the intra-block index, `a = 1` for the
diagonal-block encoding.

Two-register layout (Sec. 2 scheme, and both tidal scripts):

| register | width | meaning |
|---|---|---|
| `idx`  | $m$ | dictionary key |
| `del0` | 1 | validity flag written by $O_c$ |
| `del1` | 1 | validity flag written by $O_r$ |
| `row`  | $t$ | working register, $O_c$ writes $c_l(j)$ |
| `col`  | $t$ | block column $j$ on input, block row $i$ on output |
| `anc`  | 1 | ancilla of the diagonal-block encoding |
| `sys`  | $r$ | intra-block index $k$ |

In-place layout (Sec. 4 scheme, wave equation):

| register | width | meaning |
|---|---|---|
| `idx`  | $m$ | dictionary key |
| `del0` | 1 | validity flag written by $O_c$ |
| `col`  | $t$ | block column, overwritten in place by the block row |
| `anc`  | 1 | ancilla of the diagonal-block encoding |
| `sys`  | $r$ | intra-block index $k$ |

The multiplexed rotation $U=\sum_c|c\rangle\langle c|\otimes R_c$ is implemented
with `UCRY`/`UCRZ` gates, and the index oracles with multi-controlled NOT gates
whose controls are chosen from the key and block-position registers.

---

## 7. Implementation notes

1. **Extracting $\langle 0|U|0\rangle$.**  The matrix element is read off one
   column of $U$ at a time: the column index is prepared as the input state, the
   circuit is simulated with `AerSimulator(method="statevector")`, and the
   amplitude of the corresponding basis state is read out.  The cost therefore
   grows linearly with the number of columns; for the $256\times256$ cases the
   scripts sample a subset of columns (pass `cols=None` to `verify` to check all
   of them, which takes much longer).
2. **Do not call `transpile()`.**  With Qiskit 2.5 the high-level synthesis of
   multi-controlled gates can silently produce the wrong circuit.  The scripts
   call `QuantumCircuit.decompose()` and hand the decomposed circuit to Aer.
3. **No reset in the state preparation.**  The layered $R_y$ construction used
   for PREP/UNPREP is unitary, unlike `StatePreparation`, so the whole circuit
   can be simulated coherently.
4. **Control qubits of $U$.**  For the two Sec. 2 scripts the implementation
   selects a minimal subset of block-position bits that suffices to address the
   distinct rotations; this is an implementation detail and does not change the
   asymptotic bounds of the paper.
5. **Tidal reduction.**  Reducing to a single zonal wavenumber is what makes
   every tidal block diagonal, hence encodable by the diagonal-block encoding.
   The full two-dimensional finite-difference operator can be restored by
   replacing the mode reduction in `build_tidal` by the Kronecker structure of
   $D_\lambda$; the number of blocks then becomes $3N_\theta N_\lambda$ and $t$
   grows accordingly.

---

## 8. Citation

If you use this code, please cite the paper:

```
Yuhao Song, Chunlin Yang, Hongmei Yao,
Dictionary-based block-encoding for block-sparse matrices.
```

## 9. License

Released for academic use together with the paper.  Please contact the authors
before redistributing modified versions.
