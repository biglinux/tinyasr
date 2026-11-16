//! Zipformer2 (icefall, causal) encoder, stateless RNN-T decoder and joiner.
//!
//! Activations are row-major `t x channels`. One set of weights runs both ways:
//! - full context (`Model::full`): the second pass over a finished segment, no attention mask, each depthwise
//!   convolution sees the whole segment;
//! - streaming (`Stream`): 32-frame chunks at 50 Hz (64 feature frames, 640 ms) with 128 frames of left context,
//!   carrying the attention keys/values, the nonlinear-attention input and the convolution tails between chunks,
//!   as icefall's `streaming_forward` and the sherpa-onnx export do.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;
use crate::fbank::{BINS, Fbank};
use crate::kernels::{Linear, Packed, Quant, Weight, dot, exp, ln, sigmoid, tanh};
use alloc::format;

// ---------------------------------------------------------------------------------------------- weights

pub struct Tensors {
    pub cfg: BTreeMap<String, i32>,
    t: BTreeMap<String, (Vec<usize>, Data)>,
    pub pieces: Vec<String>,
}

/// A tensor as stored: f32 values, or integer codes with one scale per `group` weights of a row (TZF2).
enum Data {
    F32(Vec<f32>),
    Int { group: usize, row: usize, scale: Vec<f32>, q: Vec<i8> }, // row = padded row length (multiple of group)
}

impl Data {
    /// f32 values (rows x k, padding dropped).
    fn f32(self, dims: &[usize]) -> Vec<f32> {
        match self {
            Data::F32(v) => v,
            Data::Int { group, row, scale, q } => {
                let n = dims[0];
                let k: usize = dims[1..].iter().product();
                let mut out = Vec::with_capacity(n * k);
                for r in 0..n {
                    for c in 0..k {
                        out.push(q[r * row + c] as f32 * scale[r * (row / group) + c / group]);
                    }
                }
                out
            }
        }
    }
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        let s = self.b.get(self.p..self.p + n).ok_or("truncated model file")?;
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, &'static str> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<usize, &'static str> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]) as usize)
    }
    fn u32(&mut self) -> Result<usize, &'static str> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    }
    fn str(&mut self) -> Result<String, &'static str> {
        let n = self.u16()?;
        core::str::from_utf8(self.take(n)?).map(String::from).map_err(|_| "bad utf-8 in model file")
    }
}

impl Tensors {
    /// The `.tzf` file written by `export_tzf.py` (TZF1: f32) or `export_tzf2.py` (TZF2: integer weights).
    pub fn parse(b: &[u8]) -> Result<Tensors, &'static str> {
        let mut r = Rd { b, p: 0 };
        let v2 = match r.take(4)? {
            b"TZF1" => false,
            b"TZF2" => true,
            _ => return Err("not a tinyzipf model (TZF1/TZF2)"),
        };
        let mut cfg = BTreeMap::new();
        for _ in 0..r.u32()? {
            let k = r.str()?;
            let v = r.take(4)?;
            cfg.insert(k, i32::from_le_bytes([v[0], v[1], v[2], v[3]]));
        }
        let mut t = BTreeMap::new();
        for _ in 0..r.u32()? {
            let k = r.str()?;
            let integer = v2 && r.u8()? == 1;
            let nd = r.u8()? as usize;
            let dims = (0..nd).map(|_| r.u32()).collect::<Result<Vec<_>, _>>()?;
            let n: usize = dims.iter().product();
            if !integer {
                r.p += (4 - r.p % 4) % 4;
                let raw = r.take(4 * n)?;
                let v = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                t.insert(k, (dims, Data::F32(v)));
                continue;
            }
            let (bits, group, st) = (r.u8()?, r.u32()?, r.u8()?);
            r.p += (4 - r.p % 4) % 4;
            let rows = dims[0];
            let row = (n / rows).div_ceil(group) * group;
            let q: Vec<i8> = match bits {
                8 => r.take(rows * row)?.iter().map(|&c| c as i8).collect(),
                4 => r.take((rows * row).div_ceil(2))?.iter().flat_map(|&c| [(c & 15) as i8 - 8, (c >> 4) as i8 - 8]).take(rows * row).collect(),
                _ => return Err("unsupported integer width in model file"),
            };
            r.p += (4 - r.p % 4) % 4;
            let ns = rows * row / group;
            let scale = if st == 1 {
                r.take(2 * ns)?.chunks_exact(2).map(|c| f16(u16::from_le_bytes([c[0], c[1]]))).collect()
            } else {
                r.take(4 * ns)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            t.insert(k, (dims, Data::Int { group, row, scale, q }));
        }
        let pieces = (0..r.u32()?).map(|_| r.str()).collect::<Result<Vec<_>, _>>()?;
        Ok(Tensors { cfg, t, pieces })
    }

    pub fn c(&self, k: &str) -> usize {
        self.cfg[k] as usize
    }

    fn get(&mut self, k: &str) -> (Vec<usize>, Vec<f32>) {
        let (d, v) = self.t.remove(k).unwrap_or_else(|| panic!("missing tensor {k}"));
        let v = v.f32(&d);
        (d, v)
    }

    fn vec(&mut self, k: &str) -> Vec<f32> {
        self.get(k).1
    }

    fn scalar(&mut self, k: &str) -> f32 {
        self.get(k).1[0]
    }

    fn has(&self, k: &str) -> bool {
        self.t.contains_key(k)
    }

    /// Integer weights stay integer (`Quant`) when the groups tile the row exactly; otherwise f32.
    fn linear(&mut self, k: &str, bias: bool) -> Linear {
        let b = if bias { Some(alloc::format!("{k}.bias")) } else { None };
        self.linear_named(&alloc::format!("{k}.weight"), b.as_deref())
    }

    /// As `linear`, for tensors not named X.weight / X.bias (the LSTM's weight_ih_l0 ...).
    fn linear_named(&mut self, name: &str, bias: Option<&str>) -> Linear {
        let (d, data) = self.t.remove(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let (n, kk): (usize, usize) = (d[0], d[1..].iter().product());
        let b = bias.map(|k| self.vec(k));
        let w = match data {
            Data::Int { group, row, scale, q } if row == kk => Weight::Int(Quant::new(n, kk, group, scale, q)),
            data => Weight::F32(Packed::from_rows(&data.f32(&d), n, kk)),
        };
        Linear { w, b }
    }
}

// ---------------------------------------------------------------------------------------------- pieces

/// log(1 + e^x), stable.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + ln(1.0 + exp(-x.abs()))
}

fn swoosh_l(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = softplus(*v - 4.0) - 0.08 * *v - 0.035);
}

fn swoosh_r(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = softplus(*v - 1.0) - 0.08 * *v - 0.313_261_7);
}

/// BiasNorm: x * exp(log_scale) / rms(x - bias), per row.
struct BiasNorm {
    bias: Vec<f32>,
    scale: f32,
}

impl BiasNorm {
    fn load(w: &mut Tensors, k: &str) -> BiasNorm {
        BiasNorm { bias: w.vec(&alloc::format!("{k}.bias")), scale: exp(w.scalar(&alloc::format!("{k}.log_scale"))) }
    }
    fn apply(&self, x: &mut [f32]) {
        let d = self.bias.len();
        for r in x.chunks_exact_mut(d) {
            let ms = r.iter().zip(&self.bias).map(|(v, b)| (v - b) * (v - b)).sum::<f32>() / d as f32;
            let s = self.scale / ms.sqrt();
            r.iter_mut().for_each(|v| *v *= s);
        }
    }
}

/// orig + (x - orig) * scale, per channel, into x.
fn bypass(orig: &[f32], x: &mut [f32], scale: &[f32]) {
    let d = scale.len();
    for (xr, or) in x.chunks_exact_mut(d).zip(orig.chunks_exact(d)) {
        for ((v, o), s) in xr.iter_mut().zip(or).zip(scale) {
            *v = o + (*v - o) * s;
        }
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// Compact relative positional encoding (CompactRelPositionalEncoding) of relative position v.
fn rel_pe(v: f32, dim: usize, out: &mut [f32]) {
    let cl = (dim as f32).sqrt();
    let xc = cl * v.signum() * ((v.abs() + cl).ln() - cl.ln());
    let ls = dim as f32 / (2.0 * core::f32::consts::PI);
    let a = (xc / ls).atan();
    for f in 0..dim / 2 {
        let (s, c) = (a * (f + 1) as f32).sin_cos();
        out[2 * f] = c;
        out[2 * f + 1] = s;
    }
    out[dim - 1] = 1.0;
}

struct FeedForward {
    inp: Linear,
    out: Linear,
}

impl FeedForward {
    fn apply(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut h = self.inp.apply(x, t);
        swoosh_l(&mut h);
        self.out.apply(&h, t)
    }
}

/// Rows `keep` from the end of cat(cache, new) become the new cache; returns the concatenation.
fn cat_cache(cache: Option<&mut Vec<f32>>, new: Vec<f32>) -> Vec<f32> {
    match cache {
        None => new,
        Some(c) => {
            let keep = c.len();
            let mut all = core::mem::take(c);
            all.extend_from_slice(&new);
            *c = all[all.len() - keep..].to_vec();
            all
        }
    }
}

struct AttnWeights {
    in_proj: Linear,
    pos: Linear,
    heads: usize,
    qd: usize,
    pd: usize,
}

impl AttnWeights {
    /// Attention weights heads x t x (left + t). `keys`: cached keys (left x heads*qd), updated.
    /// `masked`: key slots to ignore (len left + t).
    fn apply(&self, x: &[f32], t: usize, keys: Option<&mut Vec<f32>>, left: usize, masked: &[bool], pos_dim: usize) -> Vec<f32> {
        let (h, qd, pd) = (self.heads, self.qd, self.pd);
        let y = self.in_proj.apply(x, t);
        let w = y.len() / t;
        let qdim = h * qd;
        let mut k = vec![0.0; t * qdim];
        for i in 0..t {
            k[i * qdim..(i + 1) * qdim].copy_from_slice(&y[i * w + qdim..i * w + 2 * qdim]);
        }
        let k = cat_cache(keys, k);
        let tk = left + t;
        // projected encodings of relative positions -(left + t - 1) ..= t - 1
        let np = left + 2 * t - 1;
        let mut pe = vec![0.0; np * pos_dim];
        for n in 0..np {
            rel_pe(n as f32 - (left + t - 1) as f32, pos_dim, &mut pe[n * pos_dim..(n + 1) * pos_dim]);
        }
        let p = self.pos.apply(&pe, np);
        let mut out = vec![0.0; h * t * tk];
        for hh in 0..h {
            for i in 0..t {
                let q = &y[i * w + hh * qd..i * w + (hh + 1) * qd];
                let pq = &y[i * w + 2 * qdim + hh * pd..i * w + 2 * qdim + (hh + 1) * pd];
                let row = &mut out[(hh * t + i) * tk..(hh * t + i + 1) * tk];
                for j in 0..tk {
                    row[j] = if masked.get(j).copied().unwrap_or(false) {
                        -1000.0
                    } else {
                        let n = t - 1 - i + j;
                        dot(q, &k[j * qdim + hh * qd..j * qdim + (hh + 1) * qd]) + dot(pq, &p[n * h * pd + hh * pd..n * h * pd + (hh + 1) * pd])
                    };
                }
                let m = row.iter().cloned().fold(f32::MIN, f32::max);
                let mut s = 0.0;
                row.iter_mut().for_each(|v| {
                    *v = exp(*v - m);
                    s += *v;
                });
                row.iter_mut().for_each(|v| *v /= s);
            }
        }
        out
    }
}

/// y += w * x, elementwise.
#[inline(always)]
fn axpy(y: &mut [f32], w: &[f32], x: &[f32]) {
    for ((y, w), x) in y.iter_mut().zip(w).zip(x) {
        *y += w * x;
    }
}

/// (rows x cols) -> (cols x rows).
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = x[r * cols + c];
        }
    }
    t
}

/// out (t x heads*d): per head, weights (t x tk) times values (tk x heads*d, head slice).
fn weighted(wts: &[f32], heads: usize, t: usize, tk: usize, v: &[f32], d: usize) -> Vec<f32> {
    let vd = v.len() / tk;
    let mut out = vec![0.0; t * heads * d];
    for hh in 0..heads {
        for i in 0..t {
            let o = &mut out[i * heads * d + hh * d..i * heads * d + (hh + 1) * d];
            let w = &wts[(hh * t + i) * tk..(hh * t + i + 1) * tk];
            for (j, &a) in w.iter().enumerate() {
                if a != 0.0 {
                    let vr = &v[j * vd + hh * d..j * vd + (hh + 1) * d];
                    o.iter_mut().zip(vr).for_each(|(x, y)| *x += a * y);
                }
            }
        }
    }
    out
}

struct SelfAttn {
    inp: Linear,
    out: Linear,
    heads: usize,
}

impl SelfAttn {
    fn apply(&self, x: &[f32], t: usize, wts: &[f32], tk: usize, cache: Option<&mut Vec<f32>>) -> Vec<f32> {
        let v = cat_cache(cache, self.inp.apply(x, t));
        let d = self.inp.n() / self.heads;
        self.out.apply(&weighted(wts, self.heads, t, tk, &v, d), t)
    }
}

struct NonlinAttn {
    inp: Linear,
    out: Linear,
    hid: usize,
}

impl NonlinAttn {
    /// Uses head 0 of the attention weights.
    fn apply(&self, x: &[f32], t: usize, wts: &[f32], tk: usize, cache: Option<&mut Vec<f32>>) -> Vec<f32> {
        let hid = self.hid;
        let z = self.inp.apply(x, t);
        let mut xs = vec![0.0; t * hid];
        for i in 0..t {
            let r = &z[i * 3 * hid..(i + 1) * 3 * hid];
            for c in 0..hid {
                xs[i * hid + c] = r[hid + c] * tanh(r[c]);
            }
        }
        let xs = cat_cache(cache, xs);
        let mut o = weighted(&wts[..t * tk], 1, t, tk, &xs, hid);
        for i in 0..t {
            for c in 0..hid {
                o[i * hid + c] *= z[i * 3 * hid + 2 * hid + c];
            }
        }
        self.out.apply(&o, t)
    }
}

struct Conv {
    inp: Linear,
    // weights tap-major (tap x dim), so the inner loops run over contiguous channels
    causal_w: Vec<f32>, // half x dim
    causal_b: Vec<f32>,
    chunk_w: Vec<f32>, // kernel x dim
    chunk_b: Vec<f32>,
    scale: Vec<f32>, // 2 x kernel x dim
    out: Linear,
    dim: usize,
    kernel: usize,
}

impl Conv {
    /// `cache`: the last kernel/2 rows of gated input before this chunk (streaming), updated.
    fn apply(&self, x: &[f32], t: usize, cache: Option<&mut Vec<f32>>) -> Vec<f32> {
        let (d, k) = (self.dim, self.kernel);
        let lp = k / 2;
        let z = self.inp.apply(x, t);
        let mut g = vec![0.0; t * d];
        for i in 0..t {
            for c in 0..d {
                g[i * d + c] = z[i * 2 * d + c] * sigmoid(z[i * 2 * d + d + c]);
            }
        }
        // causal part over (left pad or cache) + this input
        let mut pad = match &cache {
            Some(c) => (*c).clone(),
            None => vec![0.0; lp * d],
        };
        pad.extend_from_slice(&g);
        if let Some(c) = cache {
            *c = pad[pad.len() - lp * d..].to_vec();
        }
        let half = lp + 1;
        let mut y = vec![0.0; t * d];
        let mut b = vec![0.0; d];
        for i in 0..t {
            let a = &mut y[i * d..(i + 1) * d];
            a.copy_from_slice(&self.causal_b);
            for u in 0..half {
                axpy(a, &self.causal_w[u * d..(u + 1) * d], &pad[(i + u) * d..(i + u + 1) * d]);
            }
            // chunkwise part: centred kernel inside this chunk, zero outside, edge-scaled
            b.copy_from_slice(&self.chunk_b);
            for u in 0..k {
                let s = i as isize + u as isize - lp as isize;
                if s >= 0 && (s as usize) < t {
                    axpy(&mut b, &self.chunk_w[u * d..(u + 1) * d], &g[s as usize * d..(s as usize + 1) * d]);
                }
            }
            let (l, r) = if t < k { (Some(i), Some(k - t + i)) } else { ((i < k).then_some(i), (i >= t - k).then(|| i - (t - k))) };
            let row = |j: Option<usize>, half: usize| j.map(|j| &self.scale[(half * k + j) * d..(half * k + j + 1) * d]);
            let (le, re) = (row(l, 0), row(r, 1));
            for c in 0..d {
                let (le, re) = (le.map_or(0.0, |v| v[c]), re.map_or(0.0, |v| v[c]));
                a[c] += b[c] * (1.0 + le + re);
            }
        }
        swoosh_r(&mut y);
        self.out.apply(&y, t)
    }
}

/// Streaming state of one layer.
#[derive(Clone)]
pub struct LayerState {
    key: Vec<f32>,
    nonlin: Vec<f32>,
    val1: Vec<f32>,
    val2: Vec<f32>,
    conv1: Vec<f32>,
    conv2: Vec<f32>,
}

struct Layer {
    attn: AttnWeights,
    ff1: FeedForward,
    ff2: FeedForward,
    ff3: FeedForward,
    nonlin: NonlinAttn,
    sa1: SelfAttn,
    sa2: SelfAttn,
    conv1: Conv,
    conv2: Conv,
    norm: BiasNorm,
    bypass: Vec<f32>,
    bypass_mid: Vec<f32>,
}

impl Layer {
    fn forward(&self, x: &mut Vec<f32>, t: usize, st: Option<&mut LayerState>, left: usize, masked: &[bool], pos_dim: usize) {
        let orig = x.clone();
        let tk = left + t;
        let (mut key, mut nl, mut v1, mut v2, mut c1, mut c2) = (None, None, None, None, None, None);
        if let Some(s) = st {
            key = Some(&mut s.key);
            nl = Some(&mut s.nonlin);
            v1 = Some(&mut s.val1);
            v2 = Some(&mut s.val2);
            c1 = Some(&mut s.conv1);
            c2 = Some(&mut s.conv2);
        }
        let w = self.attn.apply(x, t, key, left, masked, pos_dim);
        let y = self.ff1.apply(x, t);
        add(x, &y);
        let y = self.nonlin.apply(x, t, &w, tk, nl);
        add(x, &y);
        let y = self.sa1.apply(x, t, &w, tk, v1);
        add(x, &y);
        let y = self.conv1.apply(x, t, c1);
        add(x, &y);
        let y = self.ff2.apply(x, t);
        add(x, &y);
        bypass(&orig, x, &self.bypass_mid);
        let y = self.sa2.apply(x, t, &w, tk, v2);
        add(x, &y);
        let y = self.conv2.apply(x, t, c2);
        add(x, &y);
        let y = self.ff3.apply(x, t);
        add(x, &y);
        self.norm.apply(x);
        bypass(&orig, x, &self.bypass);
    }
}

struct Stack {
    layers: Vec<Layer>,
    dim: usize,
    ds: usize,
    down: Vec<f32>, // softmax of the downsample bias
    combine: Vec<f32>,
}

fn softmax(b: &[f32]) -> Vec<f32> {
    let m = b.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = b.iter().map(|v| exp(v - m)).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// SimpleDownsample: weighted sum of `w.len()` consecutive rows (the last row repeated as padding).
fn downsample(x: &[f32], t: usize, d: usize, w: &[f32]) -> (Vec<f32>, usize) {
    let ds = w.len();
    let dt = t.div_ceil(ds);
    let mut out = vec![0.0; dt * d];
    for i in 0..dt {
        for (k, wk) in w.iter().enumerate() {
            let s = (i * ds + k).min(t - 1);
            out[i * d..(i + 1) * d].iter_mut().zip(&x[s * d..(s + 1) * d]).for_each(|(o, v)| *o += wk * v);
        }
    }
    (out, dt)
}

impl Stack {
    /// x: t x dim. `masked`: left-context slots at 50 Hz (len left_full) that hold no audio yet.
    fn forward(&self, x: &mut Vec<f32>, t: usize, st: Option<&mut [LayerState]>, left_full: usize, masked_full: &[bool], pos_dim: usize) {
        let d = self.dim;
        let left = left_full / self.ds;
        let mut masked: Vec<bool> = (0..left).map(|j| masked_full.get(j * self.ds).copied().unwrap_or(false)).collect();
        if self.ds == 1 {
            masked.resize(left + t, false);
            run(&self.layers, x, t, st, left, &masked, pos_dim);
            return;
        }
        let orig = x.clone();
        let (mut y, dt) = downsample(x, t, d, &self.down);
        masked.resize(left + dt, false);
        run(&self.layers, &mut y, dt, st, left, &masked, pos_dim);
        let mut up = vec![0.0; t * d];
        for i in 0..t {
            up[i * d..(i + 1) * d].copy_from_slice(&y[(i / self.ds) * d..(i / self.ds + 1) * d]);
        }
        bypass(&orig, &mut up, &self.combine);
        *x = up;
    }
}

fn run(layers: &[Layer], x: &mut Vec<f32>, t: usize, st: Option<&mut [LayerState]>, left: usize, masked: &[bool], pos_dim: usize) {
    match st {
        Some(s) => layers.iter().zip(s.iter_mut()).for_each(|(l, s)| l.forward(x, t, Some(s), left, masked, pos_dim)),
        None => layers.iter().for_each(|l| l.forward(x, t, None, left, masked, pos_dim)),
    }
}

// ---------------------------------------------------------------------------------------------- front end

/// Conv2dSubsampling + ConvNeXt, channels-last images (t x f x c).
struct Embed {
    c0: Linear, // 1 -> 8, 3x3 (im2col)
    c1: Linear, // 8 -> 32, 3x3 stride 2
    c2: Linear, // 32 -> 128, 3x3 stride (1, 2)
    dw_w: Vec<f32>, // 7 x 7 x 128 (tap-major)
    dw_b: Vec<f32>,
    pw1: Linear,
    pw2: Linear,
    out: Linear, // input order (f, c) (permuted at load from PyTorch's (c, f))
    norm: BiasNorm,
}

const FW: usize = 19; // frequency width after the convolutions
const C3: usize = 128;

/// 3x3 convolution of a channels-last image (t x f x cin), no time padding, `pf` frequency padding.
fn conv3(x: &[f32], t: usize, f: usize, cin: usize, lin: &Linear, st: usize, sf: usize, pf: usize) -> (Vec<f32>, usize, usize) {
    let to = (t - 3) / st + 1;
    let fo = (f + 2 * pf - 3) / sf + 1;
    let mut col = vec![0.0; to * fo * 9 * cin];
    for i in 0..to {
        for j in 0..fo {
            let row = &mut col[(i * fo + j) * 9 * cin..(i * fo + j + 1) * 9 * cin];
            for ci in 0..cin {
                for a in 0..3 {
                    for b in 0..3 {
                        let (tt, ff) = (i * st + a, (j * sf + b) as isize - pf as isize);
                        // PyTorch weight layout (cout, cin, 3, 3) -> column index ci*9 + a*3 + b
                        row[ci * 9 + a * 3 + b] = if ff >= 0 && (ff as usize) < f { x[(tt * f + ff as usize) * cin + ci] } else { 0.0 };
                    }
                }
            }
        }
    }
    let mut y = lin.apply(&col, to * fo);
    swoosh_r(&mut y);
    (y, to, fo)
}

impl Embed {
    /// Convolutions up to (not including) ConvNeXt: features (t x 80) -> (t' x 19 x 128), t' = (t - 7) / 2 + 1.
    fn convs(&self, feats: &[f32], t: usize) -> (Vec<f32>, usize) {
        let (x, t, f) = conv3(feats, t, 80, 1, &self.c0, 1, 1, 1);
        let (x, t, f) = conv3(&x, t, f, 8, &self.c1, 2, 2, 0);
        let (x, t, _) = conv3(&x, t, f, 32, &self.c2, 1, 2, 0);
        (x, t)
    }

    /// ConvNeXt over `rows` (t_in x 19 x 128) giving rows `[3, t_in - 3)` of the time-padded result: the caller
    /// supplies 3 rows of context (zeros or cache) on each side.
    fn convnext(&self, rows: &[f32], t_in: usize) -> Vec<f32> {
        let t = t_in - 6;
        let row = FW * C3;
        let mut y = vec![0.0; t * row];
        for i in 0..t {
            for f in 0..FW {
                let a = &mut y[i * row + f * C3..i * row + (f + 1) * C3];
                a.copy_from_slice(&self.dw_b);
                for dt in 0..7 {
                    for df in 0..7 {
                        let ff = f as isize + df as isize - 3;
                        if ff >= 0 && (ff as usize) < FW {
                            let x = &rows[(i + dt) * row + ff as usize * C3..(i + dt) * row + (ff as usize + 1) * C3];
                            axpy(a, &self.dw_w[(dt * 7 + df) * C3..(dt * 7 + df + 1) * C3], x);
                        }
                    }
                }
            }
        }
        let mut h = self.pw1.apply(&y, t * FW);
        swoosh_l(&mut h);
        let mut z = self.pw2.apply(&h, t * FW);
        add(&mut z, &rows[3 * row..(3 + t) * row]);
        z
    }

    fn finish(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = self.out.apply(x, t);
        self.norm.apply(&mut y);
        y
    }
}

// ---------------------------------------------------------------------------------------------- model

/// A recognized token: vocabulary id, output frame (40 ms) and log-probability.
#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub id: u32,
    pub frame: u32,
    pub logp: f32,
}


struct Joiner {
    enc: Linear,
    dec: Linear,
    out: Vec<(Linear, Vec<u32>)>, // per language: output rows of the allowed ids (shared + English + own), the ids
}

/// End-of-turn head (eou_head.py): P(the speaker has finished) from the second pass's last `HEAD_K` encoder frames
/// (attention pooled + the last one) and the decoder output after the last text tokens. Optional in the file.
pub const HEAD_K: usize = 50;
struct Head {
    e: Linear,
    pos: Vec<f32>, // HEAD_K x hidden
    att: Linear,
    d: Linear,
    out0: Linear,
    out2: Linear,
}

/// erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7), for the exact GELU the head was trained with.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * exp(-x * x);
    if x < 0.0 { -y } else { y }
}

fn gelu(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = 0.5 * *v * (1.0 + erf(*v * core::f32::consts::FRAC_1_SQRT_2)));
}

impl Head {
    fn load(w: &mut Tensors) -> Option<Head> {
        if !w.t.contains_key("eou_head.e.0.weight") {
            return None;
        }
        Some(Head {
            e: w.linear("eou_head.e.0", true),
            pos: w.vec("eou_head.pos"),
            att: w.linear("eou_head.att", true),
            d: w.linear("eou_head.d.0", true),
            out0: w.linear("eou_head.out.0", true),
            out2: w.linear("eou_head.out.2", true),
        })
    }

    /// enc: HEAD_K x 512 right-aligned (the first HEAD_K - valid rows unused), dec: decoder output before the joiner.
    fn prob(&self, enc: &[f32], valid: usize, dec: &[f32]) -> f32 {
        let k = HEAD_K;
        let valid = valid.clamp(1, k);
        let d = enc.len() / k;
        let mut h = self.e.apply(&enc[(k - valid) * d..], valid);
        gelu(&mut h);
        let hd = h.len() / valid;
        for (i, row) in h.chunks_exact_mut(hd).enumerate() {
            let p = &self.pos[(k - valid + i) * hd..(k - valid + i + 1) * hd];
            row.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        let sc = self.att.apply(&h, valid);
        let m = sc.iter().cloned().fold(f32::MIN, f32::max);
        let wts: Vec<f32> = sc.iter().map(|v| exp(v - m)).collect();
        let z: f32 = wts.iter().sum();
        let mut x = vec![0.0; 2 * hd];
        for (i, row) in h.chunks_exact(hd).enumerate() {
            row.iter().enumerate().for_each(|(j, v)| x[j] += wts[i] / z * v);
        }
        x[hd..].copy_from_slice(&h[(valid - 1) * hd..]);
        let mut dd = self.d.apply(dec, 1);
        gelu(&mut dd);
        x.extend_from_slice(&dd);
        let mut y = self.out0.apply(&x, 1);
        gelu(&mut y);
        sigmoid(self.out2.apply(&y, 1)[0])
    }
}

/// v5.9t hybrid decoder: an LSTM over the whole token history, added to the stateless output through a scalar gate
/// (decoder.py: embedding_out + gate * lstm_proj(lstm(embedding(y)))). One step per emitted token.
struct Lstm {
    ih: Linear, // 4H x D, bias = b_ih + b_hh
    hh: Linear, // 4H x H
    proj: Linear, // D x H
    gate: f32,
    hidden: usize,
}

impl Lstm {
    /// One step of torch.nn.LSTM (gate order i, f, g, o) on input x; updates h, c.
    fn step(&self, x: &[f32], h: &mut [f32], c: &mut [f32]) {
        let n = self.hidden;
        let mut g = self.ih.apply(x, 1);
        let r = self.hh.apply(h, 1);
        g.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
        for j in 0..n {
            let (i, f, gg, o) = (sigmoid(g[j]), sigmoid(g[n + j]), tanh(g[2 * n + j]), sigmoid(g[3 * n + j]));
            c[j] = f * c[j] + i * gg;
            h[j] = o * tanh(c[j]);
        }
    }
}

/// Most zero-duration tokens on one frame in TDT greedy before it is forced on (the server scorer's max_sym).
const TDT_MAX_SYM: usize = 5;

/// One beam-search hypothesis: decoder state, total log-probability, tokens.
#[derive(Clone)]
pub struct Hyp {
    dec: Decoding,
    logp: f32,
    toks: Vec<Token>,
    pending: bool, // TDT beam: the last token is not yet in `dec` (advanced only if the hypothesis survives pruning)
}

pub struct Model {
    head: Option<Head>,
    embed: Embed,
    stacks: Vec<Stack>,
    out_down: Vec<f32>,
    lang_vec: Vec<Vec<f32>>,
    dec_emb: Vec<f32>,
    dec_conv: Vec<f32>, // dim x 4 x context (groups of 4 channels)
    dec_dim: usize,
    context: usize,
    joiner: Joiner,
    lstm: Option<Lstm>,
    dur: Option<Linear>, // TDT duration head (durations 0..n-1) on the joiner hidden; None = RNN-T
    pos_dim: usize,
    pub left: usize,
    pub chunk: usize,
    pub pieces: Vec<String>,
    /// The languages, in the model's order (its lang_embed rows and output blocks).
    pub langs: Vec<crate::Lang>,
}

impl Model {
    /// `l`'s place among the model's languages (its first, English, for one it doesn't have).
    pub fn lang_index(&self, l: crate::Lang) -> usize {
        self.langs.iter().position(|&m| m == l).unwrap_or(0)
    }
}

/// Most tokens emitted on one encoder frame in greedy search (from the MAX_SYMBOLS env var in std builds).
#[cfg(feature = "std")]
static MAX_SYMBOLS_ENV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
#[cfg(feature = "std")]
fn max_symbols() -> usize {
    *MAX_SYMBOLS_ENV.get_or_init(|| std::env::var("MAX_SYMBOLS").ok().and_then(|v| v.parse().ok()).unwrap_or(3))
}
#[cfg(not(feature = "std"))]
fn max_symbols() -> usize {
    3
}

impl Model {
    pub fn from_bytes(b: &[u8]) -> Result<Model, &'static str> {
        let mut w = Tensors::parse(b)?;
        let ns = w.c("n_stacks");
        let pos_dim = w.c("pos_dim");
        let mut stacks = Vec::new();
        for s in 0..ns {
            let g = |k: &str| w.c(&format!("{k}.{s}"));
            let (dim, nl, ds, heads, qd, pd, kernel) =
                (g("encoder_dim"), g("num_encoder_layers"), g("downsampling_factor"), g("num_heads"), g("query_head_dim"), g("pos_head_dim"), g("cnn_module_kernel"));
            let base = if ds == 1 { format!("encoder.encoders.{s}") } else { format!("encoder.encoders.{s}.encoder") };
            let mut layers = Vec::new();
            for l in 0..nl {
                let p = format!("{base}.layers.{l}");
                let ff = |w: &mut Tensors, n: &str| FeedForward { inp: w.linear(&format!("{p}.{n}.in_proj"), true), out: w.linear(&format!("{p}.{n}.out_proj"), true) };
                let conv = |w: &mut Tensors, n: &str| Conv {
                    inp: w.linear(&format!("{p}.{n}.in_proj"), true),
                    causal_w: transpose(&w.vec(&format!("{p}.{n}.depthwise_conv.causal_conv.weight")), dim, kernel / 2 + 1),
                    causal_b: w.vec(&format!("{p}.{n}.depthwise_conv.causal_conv.bias")),
                    chunk_w: transpose(&w.vec(&format!("{p}.{n}.depthwise_conv.chunkwise_conv.weight")), dim, kernel),
                    chunk_b: w.vec(&format!("{p}.{n}.depthwise_conv.chunkwise_conv.bias")),
                    scale: {
                        let sc = w.vec(&format!("{p}.{n}.depthwise_conv.chunkwise_conv_scale"));
                        let (l, r) = sc.split_at(dim * kernel);
                        let mut t = transpose(l, dim, kernel);
                        t.extend(transpose(r, dim, kernel));
                        t
                    },
                    out: w.linear(&format!("{p}.{n}.out_proj"), true),
                    dim,
                    kernel,
                };
                let sa = |w: &mut Tensors, n: &str| SelfAttn { inp: w.linear(&format!("{p}.{n}.in_proj"), true), out: w.linear(&format!("{p}.{n}.out_proj"), true), heads };
                layers.push(Layer {
                    attn: AttnWeights { in_proj: w.linear(&format!("{p}.self_attn_weights.in_proj"), true), pos: w.linear(&format!("{p}.self_attn_weights.linear_pos"), false), heads, qd, pd },
                    ff1: ff(&mut w, "feed_forward1"),
                    ff2: ff(&mut w, "feed_forward2"),
                    ff3: ff(&mut w, "feed_forward3"),
                    nonlin: NonlinAttn { inp: w.linear(&format!("{p}.nonlin_attention.in_proj"), true), out: w.linear(&format!("{p}.nonlin_attention.out_proj"), true), hid: 3 * dim / 4 },
                    sa1: sa(&mut w, "self_attn1"),
                    sa2: sa(&mut w, "self_attn2"),
                    conv1: conv(&mut w, "conv_module1"),
                    conv2: conv(&mut w, "conv_module2"),
                    norm: BiasNorm::load(&mut w, &format!("{p}.norm")),
                    bypass: w.vec(&format!("{p}.bypass.bypass_scale")),
                    bypass_mid: w.vec(&format!("{p}.bypass_mid.bypass_scale")),
                });
            }
            let (down, combine) = if ds == 1 {
                (vec![], vec![])
            } else {
                (softmax(&w.vec(&format!("encoder.encoders.{s}.downsample.bias"))), w.vec(&format!("encoder.encoders.{s}.out_combiner.bypass_scale")))
            };
            stacks.push(Stack { layers, dim, ds, down, combine });
        }
        let out_down = softmax(&w.vec("encoder.downsample_output.bias"));
        // front end: im2col weights stay PyTorch (cout, cin*9); `out` takes (f, c) instead of (c, f)
        let (od, ow) = w.get("encoder_embed.out.weight");
        let mut ow2 = vec![0.0; ow.len()];
        for n in 0..od[0] {
            for c in 0..C3 {
                for f in 0..FW {
                    ow2[n * od[1] + f * C3 + c] = ow[n * od[1] + c * FW + f];
                }
            }
        }
        let embed = Embed {
            c0: w.linear("encoder_embed.conv.0", true),
            c1: w.linear("encoder_embed.conv.4", true),
            c2: w.linear("encoder_embed.conv.7", true),
            dw_w: transpose(&w.vec("encoder_embed.convnext.depthwise_conv.weight"), C3, 49),
            dw_b: w.vec("encoder_embed.convnext.depthwise_conv.bias"),
            pw1: w.linear("encoder_embed.convnext.pointwise_conv1", true),
            pw2: w.linear("encoder_embed.convnext.pointwise_conv2", true),
            out: Linear { w: Weight::F32(Packed::from_rows(&ow2, od[0], od[1])), b: Some(w.vec("encoder_embed.out.bias")) },
            norm: BiasNorm::load(&mut w, "encoder_embed.out_norm"),
        };
        use crate::Lang;
        // the language set: "lang.K" = Lang index of the K-th language (files from en-ru-uz on); older files by its size
        let n = w.c("n_langs");
        let langs = if w.cfg.contains_key("lang.0") {
            let mut v = alloc::vec::Vec::new();
            for k in 0..n {
                let i = *w.cfg.get(&alloc::format!("lang.{k}")).ok_or("missing language code")? as usize;
                v.push(*Lang::ALL.get(i).ok_or("unknown language code")?);
            }
            v
        } else {
            match n {
                3 => alloc::vec![Lang::En, Lang::Ru, Lang::Uk],
                4 => alloc::vec![Lang::En, Lang::Es, Lang::It, Lang::Pt],
                _ => return Err("unknown language set"),
            }
        };
        let le = w.vec("lang_embed.weight");
        let d0 = stacks[0].dim;
        let lang_vec = (0..langs.len()).map(|k| le[k * d0..(k + 1) * d0].to_vec()).collect();
        let (shared, size) = (w.c("shared") as u32, w.c("lang_size") as u32);
        let (jd, jw) = w.get("joiner.output_linear.weight");
        let jb = w.vec("joiner.output_linear.bias");
        let out = (0..langs.len() as u32)
            .map(|k| {
                // shared block + English; ґ pieces in the shared block (reserved ids, tok_dir/extra.json) only for Ukrainian
                let uk_only = |i: u32| i < shared && w.pieces.get(i as usize).is_some_and(|p| p.contains('ґ') || p.contains('Ґ'));
                let mut ids: Vec<u32> = (0..shared + size).filter(|&i| langs[k as usize] == Lang::Uk || !uk_only(i)).collect();
                if k != 0 {
                    ids.extend(shared + k * size..shared + (k + 1) * size);
                }
                let mut rows = Vec::with_capacity(ids.len() * jd[1]);
                for &i in &ids {
                    rows.extend_from_slice(&jw[i as usize * jd[1]..(i as usize + 1) * jd[1]]);
                }
                let b = ids.iter().map(|&i| jb[i as usize]).collect();
                (Linear { w: Weight::F32(Packed::from_rows(&rows, ids.len(), jd[1])), b: Some(b) }, ids)
            })
            .collect();
        let joiner = Joiner { enc: w.linear("joiner.encoder_proj", true), dec: w.linear("joiner.decoder_proj", true), out };
        let dur = if w.has("joiner.duration_linear.weight") { Some(w.linear("joiner.duration_linear", true)) } else { None };
        let lstm = if w.has("decoder.lstm.weight_ih_l0") {
            let mut ih = w.linear_named("decoder.lstm.weight_ih_l0", None);
            let (b1, b2) = (w.vec("decoder.lstm.bias_ih_l0"), w.vec("decoder.lstm.bias_hh_l0"));
            ih.b = Some(b1.iter().zip(&b2).map(|(a, b)| a + b).collect());
            let hh = w.linear_named("decoder.lstm.weight_hh_l0", None);
            let proj = w.linear_named("decoder.lstm_proj.weight", None);
            let hidden = b2.len() / 4;
            Some(Lstm { ih, hh, proj, gate: w.scalar("decoder.lstm_gate_m.g"), hidden })
        } else {
            None
        };
        let (dd, dec_emb) = w.get("decoder.embedding.weight");
        let head = Head::load(&mut w);
        Ok(Model {
            head,
            embed,
            stacks,
            out_down,
            lang_vec,
            dec_emb,
            dec_conv: w.vec("decoder.conv.weight"),
            dec_dim: dd[1],
            context: w.c("context"),
            joiner,
            lstm,
            dur,
            pos_dim,
            left: w.c("left_context"),
            chunk: w.c("chunk"),
            pieces: core::mem::take(&mut w.pieces),
            langs,
        })
    }

    /// Stacks + output combination: x (t x dim0) at 50 Hz -> (ceil(t / 2) x 512) at 25 Hz.
    fn encode(&self, x: Vec<f32>, t: usize, mut states: Option<&mut [Vec<LayerState>]>, masked: &[bool]) -> (Vec<f32>, usize) {
        let mut outs: Vec<Vec<f32>> = Vec::new();
        let mut cur = x;
        let mut cd = self.stacks[0].dim;
        let left = if states.is_some() { self.left } else { 0 };
        for (s, st) in self.stacks.iter().enumerate() {
            if st.dim != cd {
                // convert_num_channels: truncate or zero-pad to this stack's width
                let mut y = vec![0.0; t * st.dim];
                let m = st.dim.min(cd);
                for i in 0..t {
                    y[i * st.dim..i * st.dim + m].copy_from_slice(&cur[i * cd..i * cd + m]);
                }
                cur = y;
                cd = st.dim;
            }
            let ls = states.as_deref_mut().map(|all| &mut all[s][..]);
            st.forward(&mut cur, t, ls, left, masked, self.pos_dim);
            outs.push(cur.clone());
        }
        // full-dim output: the last stack, then the extra channels of earlier, wider stacks
        let last = self.stacks.len() - 1;
        let mut pieces = vec![(last, 0, self.stacks[last].dim)];
        let mut od = self.stacks[last].dim;
        for i in (0..last).rev() {
            let d = self.stacks[i].dim;
            if d > od {
                pieces.push((i, od, d));
                od = d;
            }
        }
        let mut full = vec![0.0; t * od];
        for i in 0..t {
            let mut o = 0;
            for &(s, a, b) in &pieces {
                let sd = self.stacks[s].dim;
                full[i * od + o..i * od + o + b - a].copy_from_slice(&outs[s][i * sd + a..i * sd + b]);
                o += b - a;
            }
        }
        downsample(&full, t, od, &self.out_down)
    }

    /// Full-context encoder output (frames at 25 Hz x 512) for fbank features (t x 80).
    pub fn encode_full(&self, feats: &[f32], t: usize, lang: usize) -> (Vec<f32>, usize) {
        let (x, tc) = self.embed.convs(feats, t);
        let row = FW * C3;
        let mut rows = vec![0.0; (tc + 6) * row];
        rows[3 * row..(3 + tc) * row].copy_from_slice(&x);
        let x = self.embed.convnext(&rows, tc + 6);
        let mut e = self.embed.finish(&x, tc);
        let lv = &self.lang_vec[lang];
        e.chunks_exact_mut(lv.len()).for_each(|r| add(r, lv));
        self.encode(e, tc, None, &[])
    }

    fn decoder_out(&self, ctx: &[u32], h: &[f32]) -> Vec<f32> {
        let mut y = self.decoder_y(ctx);
        if let Some(l) = &self.lstm {
            let p = l.proj.apply(h, 1);
            y.iter_mut().zip(&p).for_each(|(a, b)| *a += l.gate * b);
        }
        self.joiner.dec.apply(&y, 1)
    }

    /// `advance` for several hypotheses at once: the LSTM, its projection and the joiner's decoder projection run as
    /// one batch.
    fn advance_batch(&self, decs: &mut [&mut Decoding], ids: &[u32]) {
        let n = decs.len();
        if n == 0 {
            return;
        }
        let d = self.dec_dim;
        let mut ys: Vec<f32> = Vec::with_capacity(n * d);
        for (dec, &id) in decs.iter_mut().zip(ids) {
            dec.ctx.push(id);
            ys.extend(self.decoder_y(&dec.ctx));
        }
        if let Some(l) = &self.lstm {
            let hn = l.hidden;
            let mut x = Vec::with_capacity(n * d);
            let mut hp = Vec::with_capacity(n * hn);
            for (dec, &id) in decs.iter().zip(ids) {
                x.extend_from_slice(&self.dec_emb[id as usize * d..(id as usize + 1) * d]);
                hp.extend_from_slice(&dec.h);
            }
            let mut g = l.ih.apply(&x, n);
            let r = l.hh.apply(&hp, n);
            g.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
            for (b, dec) in decs.iter_mut().enumerate() {
                let g = &g[b * 4 * hn..(b + 1) * 4 * hn];
                for j in 0..hn {
                    let (i, f, gg, o) = (sigmoid(g[j]), sigmoid(g[hn + j]), tanh(g[2 * hn + j]), sigmoid(g[3 * hn + j]));
                    dec.c[j] = f * dec.c[j] + i * gg;
                    dec.h[j] = o * tanh(dec.c[j]);
                }
            }
            let mut hs = Vec::with_capacity(n * hn);
            decs.iter().for_each(|dec| hs.extend_from_slice(&dec.h));
            let p = l.proj.apply(&hs, n);
            ys.iter_mut().zip(&p).for_each(|(a, b)| *a += l.gate * b);
        }
        let o = self.joiner.dec.apply(&ys, n);
        let jd = o.len() / n;
        for (b, dec) in decs.iter_mut().enumerate() {
            dec.out = o[b * jd..(b + 1) * jd].to_vec();
        }
    }

    /// Joiner hidden of every hypothesis at encoder frame i (rows), from the projected encoder output `ep`.
    fn joint_rows(&self, ep: &[f32], i: usize, decs: &[&Decoding]) -> Vec<f32> {
        let jd = self.joiner.enc.n();
        let mut h = Vec::with_capacity(decs.len() * jd);
        for dec in decs {
            h.extend((0..jd).map(|k| tanh(ep[i * jd + k] + dec.out[k])));
        }
        h
    }

    /// Emit `id`: extend the history, advance the LSTM, recompute the decoder output.
    fn advance(&self, dec: &mut Decoding, id: u32) {
        dec.ctx.push(id);
        if let Some(l) = &self.lstm {
            let d = self.dec_dim;
            l.step(&self.dec_emb[id as usize * d..(id as usize + 1) * d], &mut dec.h, &mut dec.c);
        }
        dec.out = self.decoder_out(&dec.ctx, &dec.h);
    }


    /// Decoder output before the joiner's projection (dec_dim), for the last `context` tokens of `ctx`.
    fn decoder_y(&self, ctx: &[u32]) -> Vec<f32> {
        let d = self.dec_dim;
        let c = self.context;
        let mut y = vec![0.0; d];
        for o in 0..d {
            let g = o / 4;
            let mut a = 0.0;
            for i in 0..4 {
                for (k, &tok) in ctx[ctx.len() - c..].iter().enumerate() {
                    a += self.dec_conv[(o * 4 + i) * c + k] * self.dec_emb[tok as usize * d + g * 4 + i];
                }
            }
            y[o] = a.max(0.0);
        }
        y
    }

    /// Greedy RNN-T over encoder frames, continuing from `dec`.
    fn greedy(&self, enc: &[f32], t: usize, lang: usize, dec: &mut Decoding, frame0: u32) -> Vec<Token> {
        if let Some(dur) = &self.dur {
            return self.greedy_tdt(dur, enc, t, lang, dec, frame0);
        }
        let ep = self.joiner.enc.apply(enc, t);
        let jd = self.joiner.enc.n();
        let (out, ids) = &self.joiner.out[lang];
        let mut toks = Vec::new();
        let mut h = vec![0.0; jd];
        for i in 0..t {
            // up to max_symbols() tokens per 40 ms frame (fast speech needs more than one)
            for _ in 0..max_symbols() {
                for (k, v) in h.iter_mut().enumerate() {
                    *v = tanh(ep[i * jd + k] + dec.out[k]);
                }
                let logit = out.apply(&h, 1);
                let (best, &m) = logit.iter().enumerate().fold((0, &f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
                let id = ids[best];
                if id == 0 {
                    break;
                }
                let lse = m + logit.iter().map(|v| exp(v - m)).sum::<f32>().ln();
                toks.push(Token { id, frame: frame0 + i as u32, logp: m - lse });
                self.advance(dec, id);
            }
        }
        toks
    }

    /// modified_beam_search (icefall beam_search.py): per frame, every hypothesis is extended by its log-softmax over the
    /// language's tokens; the `beam` best (hypothesis, token) pairs survive, blank keeping the text; at most one token
    /// per frame; hypotheses with the same tokens are merged (log-add). Continues from `hyps`.
    fn beam_rnnt(&self, enc: &[f32], t: usize, lang: usize, hyps: &mut Vec<Hyp>, frame0: u32, beam: usize) {
        let ep = self.joiner.enc.apply(enc, t);
        let (out, ids) = &self.joiner.out[lang];
        for i in 0..t {
            let mut cand: Vec<(f32, usize, usize, f32)> = Vec::new(); // (score, hyp, output index, token logp)
            let nh = hyps.len();
            let hrows = self.joint_rows(&ep, i, &hyps.iter().map(|h| &h.dec).collect::<Vec<_>>());
            let lg = out.apply(&hrows, nh);
            let vn = lg.len() / nh;
            for (hi, hy) in hyps.iter().enumerate() {
                let logit = &lg[hi * vn..(hi + 1) * vn];
                let m = logit.iter().cloned().fold(f32::MIN, f32::max);
                let lse = m + logit.iter().map(|v| exp(v - m)).sum::<f32>().ln();
                for (k, &v) in logit.iter().enumerate() {
                    cand.push((hy.logp + v - lse, hi, k, v - lse));
                }
            }
            let n = beam.min(cand.len());
            cand.select_nth_unstable_by(n - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            cand.truncate(n);
            let mut next: Vec<Hyp> = Vec::with_capacity(n);
            for &(score, hi, k, lp) in &cand {
                let id = ids[k];
                let mut hy = hyps[hi].clone();
                hy.logp = score;
                if id != 0 {
                    hy.toks.push(Token { id, frame: frame0 + i as u32, logp: lp });
                    hy.pending = true;
                }
                if let Some(o) = next.iter_mut().find(|o| o.toks.len() == hy.toks.len() && o.toks.iter().zip(&hy.toks).all(|(a, b)| a.id == b.id)) {
                    let (a, b) = (o.logp.max(hy.logp), o.logp.min(hy.logp));
                    o.logp = a + (1.0 + exp(b - a)).ln();
                } else {
                    next.push(hy);
                }
            }
            let pend: Vec<u32> = next.iter().filter(|h| h.pending).map(|h| h.toks.last().unwrap().id).collect();
            let mut decs: Vec<&mut Decoding> = next.iter_mut().filter(|h| h.pending).map(|h| { h.pending = false; &mut h.dec }).collect();
            self.advance_batch(&mut decs, &pend);
            *hyps = next;
        }
    }

    fn beam_tdt(&self, dur: &Linear, enc: &[f32], t: usize, lang: usize, beam: usize) -> Vec<Token> {
        let ep = self.joiner.enc.apply(enc, t);
        let (out, ids) = &self.joiner.out[lang];
        let lsm = |v: &[f32]| {
            let m = v.iter().cloned().fold(f32::MIN, f32::max);
            let lse = m + v.iter().map(|x| exp(x - m)).sum::<f32>().ln();
            v.iter().map(|x| x - lse).collect::<Vec<f32>>()
        };
        let same = |a: &Hyp, b: &Hyp| a.toks.len() == b.toks.len() && a.toks.iter().zip(&b.toks).all(|(x, y)| x.id == y.id);
        // merge `hy` into `list` (log-add on equal text)
        let push = |list: &mut Vec<Hyp>, hy: Hyp| {
            if let Some(o) = list.iter_mut().find(|o| same(o, &hy)) {
                let (a, b) = (o.logp.max(hy.logp), o.logp.min(hy.logp));
                o.logp = a + (1.0 + exp(b - a)).ln();
            } else {
                list.push(hy);
            }
        };
        let mut buckets: Vec<Vec<Hyp>> = (0..t).map(|_| Vec::new()).collect();
        let mut done: Vec<Hyp> = Vec::new();
        if t == 0 {
            return Vec::new();
        }
        buckets[0] = self.start_beam();
        for i in 0..t {
            let mut cur = core::mem::take(&mut buckets[i]);
            for _round in 0..=TDT_MAX_SYM {
                if cur.is_empty() {
                    break;
                }
                cur.sort_by(|a, b| b.logp.partial_cmp(&a.logp).unwrap_or(core::cmp::Ordering::Equal));
                cur.truncate(beam);
                {  // decoder step for the survivors only, as one batch
                    let pend: Vec<u32> = cur.iter().filter(|h| h.pending).map(|h| h.toks.last().unwrap().id).collect();
                    let mut decs: Vec<&mut Decoding> = cur.iter_mut().filter(|h| h.pending).map(|h| { h.pending = false; &mut h.dec }).collect();
                    self.advance_batch(&mut decs, &pend);
                }
                let mut stay: Vec<Hyp> = Vec::new(); // zero-duration tokens: same frame, next round
                let nh = cur.len();
                let hrows = self.joint_rows(&ep, i, &cur.iter().map(|h| &h.dec).collect::<Vec<_>>());
                let lgall = out.apply(&hrows, nh);
                let ldall = dur.apply(&hrows, nh);
                let (vn, dn) = (lgall.len() / nh, ldall.len() / nh);
                for (hi, hy) in cur.iter().enumerate() {
                    let lt = lsm(&lgall[hi * vn..(hi + 1) * vn]);
                    let ld = lsm(&ldall[hi * dn..(hi + 1) * dn]);
                    // this hypothesis' best `beam` (token, duration) pairs
                    let mut cand: Vec<(f32, usize, usize)> = Vec::new();
                    let mut order: Vec<usize> = (0..lt.len()).collect();
                    let kk = beam.min(order.len());
                    order.select_nth_unstable_by(kk - 1, |a, b| lt[*b].partial_cmp(&lt[*a]).unwrap_or(core::cmp::Ordering::Equal));
                    for &k in &order[..kk] {
                        for (d, &l) in ld.iter().enumerate() {
                            if ids[k] == 0 && d == 0 {
                                continue;
                            }
                            cand.push((hy.logp + lt[k] + l, k, d));
                        }
                    }
                    let n = beam.min(cand.len());
                    cand.select_nth_unstable_by(n - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
                    for &(score, k, d) in &cand[..n] {
                        let id = ids[k];
                        let mut nh = hy.clone();
                        nh.logp = score;
                        if id != 0 {
                            nh.toks.push(Token { id, frame: i as u32, logp: lt[k] });
                            nh.pending = true;
                        }
                        if d == 0 {
                            push(&mut stay, nh);
                        } else if i + d >= t {
                            push(&mut done, nh);
                        } else {
                            push(&mut buckets[i + d], nh);
                        }
                    }
                }
                cur = stay;
            }
            for mut hy in cur {  // still on this frame after TDT_MAX_SYM zero-duration rounds: forced on by one frame
                if hy.pending {
                    let id = hy.toks.last().map(|x| x.id).unwrap_or(0);
                    self.advance(&mut hy.dec, id);
                    hy.pending = false;
                }
                if i + 1 >= t { push(&mut done, hy) } else { push(&mut buckets[i + 1], hy) }
            }
        }
        Model::best(&done)
    }

    fn start_beam(&self) -> Vec<Hyp> {
        vec![Hyp { dec: Decoding::new(self), logp: 0.0, toks: Vec::new(), pending: false }]
    }

    fn best(hyps: &[Hyp]) -> Vec<Token> {
        hyps.iter().max_by(|a, b| a.logp.partial_cmp(&b.logp).unwrap_or(core::cmp::Ordering::Equal)).map(|h| h.toks.clone()).unwrap_or_default()
    }

    /// TDT greedy (tdt_decode.py): after a token or blank the frame pointer moves on by the predicted duration (a blank
    /// by at least 1; after TDT_MAX_SYM zero-duration tokens in a row by 1). A jump past this chunk's last frame is
    /// carried into the next chunk (`dec.skip`).
    fn greedy_tdt(&self, dur: &Linear, enc: &[f32], t: usize, lang: usize, dec: &mut Decoding, frame0: u32) -> Vec<Token> {
        let ep = self.joiner.enc.apply(enc, t);
        let jd = self.joiner.enc.n();
        let (out, ids) = &self.joiner.out[lang];
        let mut toks = Vec::new();
        let mut h = vec![0.0; jd];
        let argmax = |v: &[f32], from: usize| (from..v.len()).fold(from, |a, j| if v[j] > v[a] { j } else { a });
        let mut i = dec.skip;
        while i < t {
            for (k, v) in h.iter_mut().enumerate() {
                *v = tanh(ep[i * jd + k] + dec.out[k]);
            }
            let logit = out.apply(&h, 1);
            let dl = dur.apply(&h, 1);
            let best = argmax(&logit, 0);
            let id = ids[best];
            let mut d;
            if id == 0 {
                d = argmax(&dl, 1); // a blank always moves on
                dec.run = 0;
            } else {
                d = argmax(&dl, 0);
                let m = logit[best];
                let lse = m + logit.iter().map(|v| exp(v - m)).sum::<f32>().ln();
                toks.push(Token { id, frame: frame0 + i as u32, logp: m - lse });
                self.advance(dec, id);
                dec.run = if d == 0 { dec.run + 1 } else { 0 };
                if dec.run >= TDT_MAX_SYM {
                    d = 1;
                    dec.run = 0;
                }
            }
            i += d;
        }
        dec.skip = i - t;
        toks
    }


    /// Second pass: full-context transcription of a finished segment (16 kHz); beam > 1: beam search of that width
    /// (RNN-T: modified_beam_search; TDT: over tokens and durations), else greedy.
    pub fn full(&self, audio: &[f32], lang: usize, beam: usize) -> Vec<Token> {
        let fb = Fbank::new();
        let t = Fbank::num_frames(audio.len());
        if t < 16 {
            return Vec::new();
        }
        let feats = fb.frames(audio, 0, t);
        let (enc, te) = self.encode_full(&feats, t, lang);
        if beam > 1 {
            if let Some(dur) = &self.dur {
                return self.beam_tdt(dur, &enc, te, lang, beam);
            }
            let mut hyps = self.start_beam();
            self.beam_rnnt(&enc, te, lang, &mut hyps, 0, beam);
            return Model::best(&hyps);
        }
        let mut dec = Decoding::new(self);
        self.greedy(&enc, te, lang, &mut dec, 0)
    }

    /// Second pass + P(the speaker has finished) from the end-of-turn head (None if the file has no head): the last
    /// HEAD_K encoder frames and the decoder output after the last text tokens (<eou>, id 12, skipped).
    pub fn full_head(&self, audio: &[f32], lang: usize) -> (Vec<Token>, Option<f32>) {
        let fb = Fbank::new();
        let t = Fbank::num_frames(audio.len());
        if t < 16 {
            return (Vec::new(), None);
        }
        let feats = fb.frames(audio, 0, t);
        let (enc, te) = self.encode_full(&feats, t, lang);
        let mut dec = Decoding::new(self);
        let toks = self.greedy(&enc, te, lang, &mut dec, 0);
        let p = self.head.as_ref().map(|h| {
            let d = enc.len() / te.max(1);
            let n = te.min(HEAD_K);
            let mut last = vec![0.0; HEAD_K * d];
            last[(HEAD_K - n) * d..].copy_from_slice(&enc[(te - n) * d..te * d]);
            let ids: Vec<u32> = toks.iter().map(|t| t.id).filter(|&i| i != 12).collect();
            let mut ctx = vec![0u32; self.context];
            ctx.extend(&ids);
            let mut y = self.decoder_y(&ctx);
            if let Some(l) = &self.lstm {  // hybrid decoder: + the LSTM output after [blank] + the text (eou_head.py)
                let dd = self.dec_dim;
                let (mut hs, mut cs) = (vec![0.0; l.hidden], vec![0.0; l.hidden]);
                for &i in [0].iter().chain(&ids) {
                    l.step(&self.dec_emb[i as usize * dd..(i as usize + 1) * dd], &mut hs, &mut cs);
                }
                let p = l.proj.apply(&hs, 1);
                y.iter_mut().zip(&p).for_each(|(a, b)| *a += l.gate * b);
            }
            h.prob(&last, n, &y)
        });
        (toks, p)
    }

    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    pub fn stream(&self, lang: usize) -> Stream<'_> {
        Stream::new(self, lang)
    }

}

/// Decoder context (last `context` tokens) and its projected output.
#[derive(Clone)]
pub struct Decoding {
    ctx: Vec<u32>,
    out: Vec<f32>,
    h: Vec<f32>, // LSTM state (empty without the LSTM)
    c: Vec<f32>,
    skip: usize, // TDT: frames of the next chunk already jumped over
    run: usize,  // TDT: zero-duration tokens in a row on the current frame
}

impl Decoding {
    fn new(m: &Model) -> Decoding {
        let ctx = vec![0; m.context];
        let n = m.lstm.as_ref().map_or(0, |l| l.hidden);
        let (mut h, mut c) = (vec![0.0; n], vec![0.0; n]);
        if let Some(l) = &m.lstm {  // training runs the LSTM over [blank] + labels: the start state has seen blank
            l.step(&m.dec_emb[..m.dec_dim], &mut h, &mut c);
        }
        let out = m.decoder_out(&ctx, &h);
        Decoding { ctx, out, h, c, skip: 0, run: 0 }
    }
}

// ---------------------------------------------------------------------------------------------- streaming

/// Feature frames per chunk window and per step: 2 * chunk + 13 (7 for the convolutions, 6 for ConvNeXt), and
/// 2 * chunk.
pub struct Stream<'m> {
    m: &'m Model,
    lang: usize,
    fb: Fbank,
    audio: Vec<f32>,
    feats: Vec<f32>, // frames computed so far
    nfeat: usize,
    chunks: usize,
    embed_cache: Vec<f32>, // 3 x 19 x 128
    states: Vec<Vec<LayerState>>,
    processed: usize, // 50 Hz frames through the encoder
    dec: Decoding,
    pub tokens: Vec<Token>,
    finished: bool,
    /// Encoder output of every chunk so far (frames x 512), kept when `keep_enc` is set (tests).
    pub keep_enc: bool,
    pub enc: Vec<f32>,
}

impl<'m> Stream<'m> {
    fn new(m: &'m Model, lang: usize) -> Stream<'m> {
        let states = m
            .stacks
            .iter()
            .map(|st| {
                let left = m.left / st.ds;
                let d = st.dim;
                let l0 = &st.layers[0];
                let s = LayerState {
                    key: vec![0.0; left * l0.attn.heads * l0.attn.qd],
                    nonlin: vec![0.0; left * l0.nonlin.hid],
                    val1: vec![0.0; left * l0.sa1.inp.n()],
                    val2: vec![0.0; left * l0.sa2.inp.n()],
                    conv1: vec![0.0; (l0.conv1.kernel / 2) * d],
                    conv2: vec![0.0; (l0.conv2.kernel / 2) * d],
                };
                vec![s; st.layers.len()]
            })
            .collect();
        Stream {
            m,
            lang,
            fb: Fbank::new(),
            audio: Vec::new(),
            feats: Vec::new(),
            nfeat: 0,
            chunks: 0,
            embed_cache: vec![0.0; 3 * FW * C3],
            states,
            processed: 0,
            dec: Decoding::new(m),
            tokens: Vec::new(),
            finished: false,
            keep_enc: false,
            enc: Vec::new(),
        }
    }

    pub fn window(&self) -> usize {
        2 * self.m.chunk + 13
    }

    /// Feed 16 kHz audio; returns the tokens decoded from the chunks it completed.
    pub fn accept(&mut self, x: &[f32]) -> Vec<Token> {
        self.audio.extend_from_slice(x);
        let ready = if self.finished { Fbank::num_frames(self.audio.len()) } else { Fbank::num_ready(self.audio.len()) };
        if ready > self.nfeat {
            let f = self.fb.frames(&self.audio, self.nfeat, ready);
            self.feats.extend_from_slice(&f);
            self.nfeat = ready;
        }
        let mut out = Vec::new();
        let (win, step) = (self.window(), 2 * self.m.chunk);
        while self.chunks * step + win <= self.nfeat {
            let s = self.chunks * step;
            let t = self.chunk(&self.feats[s * BINS..(s + win) * BINS].to_vec());
            out.extend(t);
            self.chunks += 1;
        }
        self.tokens.extend_from_slice(&out);
        out
    }

    /// End of input: pad with silence so the last audio goes through (like sherpa-onnx's tail padding).
    pub fn finish(&mut self) -> Vec<Token> {
        let pad = vec![0.0; (self.window() * 160).max(16000)];
        let mut out = self.accept(&pad);
        self.finished = true;
        out.extend(self.accept(&[]));
        out
    }

    fn chunk(&mut self, feats: &[f32]) -> Vec<Token> {
        let m = self.m;
        let win = self.window();
        let (x, tc) = m.embed.convs(feats, win);
        let row = FW * C3;
        let mut rows = core::mem::take(&mut self.embed_cache);
        rows.extend_from_slice(&x);
        let t = tc - 3; // = chunk
        // cache (3 rows) + 35 new rows: the 7x7 time context comes from the window itself (13 frames overlap)
        let y = m.embed.convnext(&rows, tc + 3);
        self.embed_cache = rows[t * row..(t + 3) * row].to_vec();
        let mut e = m.embed.finish(&y, t);
        let lv = &m.lang_vec[self.lang];
        e.chunks_exact_mut(lv.len()).for_each(|r| add(r, lv));
        let left = m.left;
        let masked: Vec<bool> = (0..left).map(|j| j + self.processed < left).collect();
        self.processed += t;
        let (enc, te) = m.encode(e, t, Some(&mut self.states), &masked);
        if self.keep_enc {
            self.enc.extend_from_slice(&enc);
        }
        let frame0 = (self.chunks * t / 2) as u32;
        m.greedy(&enc, te, self.lang, &mut self.dec, frame0)
    }
}
