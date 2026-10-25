//! The space, put into two or three dimensions to be looked at.
//!
//! Every projection ends as one position per track, `[x, y, z]` with z zero
//! in 2D and NaN for a track the projection leaves out: off the server's map,
//! or outside the sample UMAP was run on. Positions are centred and scaled
//! so most of the cloud sits within -1 and 1, which is what the cameras frame.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use stelly_core::vectors::{Vectors, unquantise};

#[derive(Clone, Debug, PartialEq)]
pub enum Projection {
    /// The server's own layout, UMAP over the whole catalogue in 2D.
    Map,
    /// The directions the weighted space varies most along.
    Pca { dims: usize },
    /// Columns of the space as the axes, one each.
    Axes { columns: [Option<usize>; 3], spread: bool },
    /// UMAP of a sample, run here.
    Umap { dims: usize, sample: usize, neighbours: usize },
}

impl Projection {
    pub fn dims(&self) -> usize {
        match self {
            Projection::Map => 2,
            Projection::Pca { dims } | Projection::Umap { dims, .. } => *dims,
            Projection::Axes { columns, .. } => 2 + columns[2].is_some() as usize,
        }
    }

    pub fn describe(&self, v: &Vectors) -> String {
        let column = |c: Option<usize>| c.map_or("-".to_string(), |c| column_name(v, c));
        match self {
            Projection::Map => "the server's map".into(),
            Projection::Pca { dims } => format!("PCA, {dims}D"),
            Projection::Axes { columns, spread } => format!(
                "{} × {}{}{}",
                column(columns[0]),
                column(columns[1]),
                columns[2].map_or(String::new(), |c| format!(" × {}", column_name(v, c))),
                if *spread { ", by rank" } else { "" }
            ),
            Projection::Umap { dims, sample, .. } => {
                format!("UMAP, {dims}D, {} tracks", (*sample).min(v.head.n))
            }
        }
    }
}

/// "block · column", as the side panel lists them.
pub fn column_name(v: &Vectors, c: usize) -> String {
    v.head
        .blocks
        .iter()
        .find(|b| (b.start..b.end).contains(&c))
        .map_or(format!("column {c}"), |b| format!("{} · {}", b.name, b.columns.get(c - b.start).map_or("?", String::as_str)))
}

/// How a long projection reports, and is told to stop.
#[derive(Clone, Default)]
pub struct Progress {
    /// Thousandths done.
    pub done: Arc<AtomicU32>,
    pub cancelled: Arc<AtomicBool>,
}

impl Progress {
    fn at(&self, fraction: f64) {
        self.done.store((fraction.clamp(0.0, 1.0) * 1000.0) as u32, Ordering::Relaxed);
    }

    fn stopped(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// Each row scaled by its block's weight, then to unit length: the server's
/// own `Space::weighted`, so neighbours here are the server's neighbours.
pub fn weighted(v: &Vectors, weights: &[f32]) -> Vec<f32> {
    let d = v.head.dims;
    let mut scale = vec![1.0f32; d];
    for (b, w) in v.head.blocks.iter().zip(weights) {
        scale[b.start..b.end].fill(*w);
    }
    let mut out = vec![0f32; v.head.n * d];
    for (i, dst) in out.chunks_exact_mut(d).enumerate() {
        let mut norm = 0f32;
        for ((x, q), s) in dst.iter_mut().zip(v.row(i)).zip(&scale) {
            *x = unquantise(*q) * s;
            norm += *x * *x;
        }
        let norm = norm.sqrt().max(1e-9);
        dst.iter_mut().for_each(|x| *x /= norm);
    }
    out
}

/// `None` when cancelled.
pub fn run(projection: &Projection, v: &Vectors, unit: &[f32], progress: &Progress) -> Option<Vec<[f32; 3]>> {
    let n = v.head.n;
    let d = v.head.dims;
    let points = match projection {
        Projection::Map => {
            let raw: Vec<[f32; 3]> = v.xy.iter().map(|p| [p[0], p[1], 0.0]).collect();
            // The canvas map has y going down; flipped, it reads the same way here.
            let mut raw = raw;
            raw.iter_mut().for_each(|p| p[1] = -p[1]);
            fit(raw, 2, true)
        }
        Projection::Pca { dims } => {
            let components = pca(unit, n, d, *dims, progress)?;
            let mean = mean(unit, n, d);
            let raw = (0..n)
                .map(|i| {
                    let row = &unit[i * d..(i + 1) * d];
                    let mut p = [0f32; 3];
                    for (k, c) in components.iter().enumerate() {
                        p[k] = row.iter().zip(c).zip(&mean).map(|((x, w), m)| (x - m) * w).sum();
                    }
                    p
                })
                .collect();
            fit(raw, *dims, true)
        }
        Projection::Axes { columns, spread } => {
            let mut raw = vec![[0f32; 3]; n];
            for (axis, column) in columns.iter().enumerate() {
                let Some(c) = *column else { continue };
                let values: Vec<f32> = (0..n).map(|i| unquantise(v.row(i)[c])).collect();
                let values = if *spread { ranks(&values) } else { values };
                raw.iter_mut().zip(values).for_each(|(p, x)| p[axis] = x);
            }
            // Each axis its own scale: they are different quantities.
            fit(raw, projection.dims(), false)
        }
        Projection::Umap { dims, sample, neighbours } => {
            let picked = sample_rows(v, (*sample).min(n));
            let mut rows = Vec::with_capacity(picked.len() * d);
            for &i in &picked {
                rows.extend_from_slice(&unit[i * d..(i + 1) * d]);
            }
            let embedding = umap::layout(&rows, picked.len(), d, *dims, *neighbours, progress)?;
            let mut raw = vec![[f32::NAN; 3]; n];
            for (&i, p) in picked.iter().zip(embedding) {
                raw[i] = p;
                if *dims == 2 {
                    raw[i][2] = 0.0;
                }
            }
            fit(raw, *dims, true)
        }
    };
    progress.at(1.0);
    Some(points)
}

/// Every `n / count`-th track by a fixed hash of its id, so the same sample
/// comes back for the same space and a bigger sample holds a smaller one.
fn sample_rows(v: &Vectors, count: usize) -> Vec<usize> {
    if count >= v.head.n {
        return (0..v.head.n).collect();
    }
    let mut keyed: Vec<(u64, usize)> = v.ids.iter().enumerate().map(|(i, id)| (mix(*id as u64), i)).collect();
    keyed.select_nth_unstable(count);
    let mut picked: Vec<usize> = keyed[..count].iter().map(|(_, i)| *i).collect();
    picked.sort_unstable();
    picked
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// Each value's rank, from -1 for the lowest to 1 for the highest: the axis
/// spread evenly, rather than piled up where most values sit.
fn ranks(values: &[f32]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut out = vec![0f32; values.len()];
    let last = (values.len().max(2) - 1) as f32;
    for (rank, i) in order.into_iter().enumerate() {
        out[i] = rank as f32 / last * 2.0 - 1.0;
    }
    out
}

/// Centred on the median and scaled so the 2nd to 98th percentile of each
/// axis spans -1 to 1; `uniform` keeps one scale for all axes, so a shape
/// keeps its proportions.
fn fit(mut points: Vec<[f32; 3]>, dims: usize, uniform: bool) -> Vec<[f32; 3]> {
    let mut centre = [0f32; 3];
    let mut half = [1f32; 3];
    for axis in 0..dims {
        let mut values: Vec<f32> = points.iter().map(|p| p[axis]).filter(|x| x.is_finite()).collect();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let at = |q: f32| values[((values.len() - 1) as f32 * q) as usize];
        centre[axis] = at(0.5);
        half[axis] = ((at(0.98) - at(0.02)) / 2.0).max(1e-6);
    }
    if uniform {
        let widest = half[..dims].iter().copied().fold(0f32, f32::max);
        half[..dims].fill(widest);
    }
    for p in points.iter_mut() {
        for axis in 0..3 {
            p[axis] = if axis < dims { (p[axis] - centre[axis]) / half[axis] } else if p[0].is_finite() { 0.0 } else { f32::NAN };
        }
    }
    points
}

fn mean(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut m = vec![0f64; d];
    for row in data.chunks_exact(d) {
        m.iter_mut().zip(row).for_each(|(m, x)| *m += *x as f64);
    }
    m.into_iter().map(|x| (x / n.max(1) as f64) as f32).collect()
}

/// The top `k` principal directions: the covariance, then power iteration
/// with deflation. The covariance is d x d, a few dozen, so the iteration is
/// nothing; the pass over the rows is the cost, shared over the cores.
fn pca(data: &[f32], n: usize, d: usize, k: usize, progress: &Progress) -> Option<Vec<Vec<f32>>> {
    let m = mean(data, n, d);
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(16);
    let chunk = n.div_ceil(threads).max(1);
    let parts: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = data
            .chunks(chunk * d)
            .map(|rows| {
                let m = &m;
                scope.spawn(move || {
                    let mut cov = vec![0f64; d * d];
                    let mut centred = vec![0f64; d];
                    for row in rows.chunks_exact(d) {
                        centred.iter_mut().zip(row.iter().zip(m)).for_each(|(c, (x, m))| *c = (*x - *m) as f64);
                        for a in 0..d {
                            let ca = centred[a];
                            let line = &mut cov[a * d..(a + 1) * d];
                            for b in a..d {
                                line[b] += ca * centred[b];
                            }
                        }
                    }
                    cov
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    if progress.stopped() {
        return None;
    }
    progress.at(0.8);
    let mut cov = vec![0f64; d * d];
    for part in parts {
        cov.iter_mut().zip(part).for_each(|(c, p)| *c += p);
    }
    for a in 0..d {
        for b in 0..a {
            cov[a * d + b] = cov[b * d + a];
        }
    }

    let mut components = Vec::new();
    for c in 0..k {
        // A fixed start, so the same space gives the same picture.
        let mut w: Vec<f64> = (0..d).map(|i| 1.0 + ((i * 31 + c * 7) % 13) as f64 / 13.0).collect();
        let mut value = 0.0;
        for _ in 0..500 {
            let mut next = vec![0f64; d];
            for a in 0..d {
                next[a] = (0..d).map(|b| cov[a * d + b] * w[b]).sum();
            }
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
            next.iter_mut().for_each(|x| *x /= norm);
            let moved: f64 = next.iter().zip(&w).map(|(a, b)| (a - b).abs()).sum();
            w = next;
            value = norm;
            if moved < 1e-10 {
                break;
            }
        }
        // Deflate, so the next iteration finds the next direction.
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] -= value * w[a] * w[b];
            }
        }
        // The sign fixed by the largest coefficient, as `build-space` does.
        let largest = w.iter().copied().fold(0f64, |m, x| if x.abs() > m.abs() { x } else { m });
        if largest < 0.0 {
            w.iter_mut().for_each(|x| *x = -*x);
        }
        components.push(w.into_iter().map(|x| x as f32).collect());
    }
    Some(components)
}

/// A port of the server's layout stage (itself a port of umap-learn), for any
/// number of output dimensions and with progress and cancelling.
mod umap {
    use super::Progress;
    use std::collections::HashMap;

    const SPREAD: f64 = 1.0;
    const MIN_DIST: f64 = 0.1;
    const NEGATIVE_SAMPLE_RATE: usize = 5;
    const SEED: u64 = 42;

    pub fn layout(data: &[f32], n: usize, d: usize, dims: usize, k: usize, progress: &Progress) -> Option<Vec<[f32; 3]>> {
        if n < 3 {
            return Some(vec![[0.0; 3]; n]);
        }
        let k = k.clamp(2, n - 1);
        let epochs = if n > 10_000 { 200 } else { 500 };
        let knn = nearest(data, n, d, k, progress)?;
        let edges = fuzzy_graph(&knn, n, k);
        let (a, b) = fit_ab(MIN_DIST, SPREAD);
        let mut embedding = initial(data, n, d, dims);
        optimise(&mut embedding, &edges, n, dims, a, b, epochs, progress)?;
        Some(
            embedding
                .chunks_exact(dims)
                .map(|p| [p[0] as f32, p[1] as f32, if dims > 2 { p[2] as f32 } else { 0.0 }])
                .collect(),
        )
    }

    /// Brute force over every core: quadratic, which is why UMAP runs on a
    /// sample. Half the bar.
    fn nearest(data: &[f32], n: usize, d: usize, k: usize, progress: &Progress) -> Option<Vec<Vec<(usize, f64)>>> {
        let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(n);
        let chunk = n.div_ceil(threads);
        let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let done = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for (t, slots) in out.chunks_mut(chunk).enumerate() {
                let done = &done;
                scope.spawn(move || {
                    let mut scores = vec![0f32; n];
                    for (offset, slot) in slots.iter_mut().enumerate() {
                        if progress.stopped() {
                            return;
                        }
                        let i = t * chunk + offset;
                        let row = &data[i * d..(i + 1) * d];
                        for (j, score) in scores.iter_mut().enumerate() {
                            *score = row.iter().zip(&data[j * d..(j + 1) * d]).map(|(a, b)| a * b).sum();
                        }
                        scores[i] = f32::NEG_INFINITY;
                        let mut order: Vec<usize> = (0..n).collect();
                        order.select_nth_unstable_by(k, |a, b| scores[*b].total_cmp(&scores[*a]));
                        order.truncate(k);
                        order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
                        *slot = order.into_iter().map(|j| (j, (1.0 - scores[j] as f64).max(0.0))).collect();
                        let rows = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if rows.is_multiple_of(256) {
                            progress.at(0.5 * rows as f64 / n as f64);
                        }
                    }
                });
            }
        });
        (!progress.stopped()).then_some(out)
    }

    fn fuzzy_graph(knn: &[Vec<(usize, f64)>], n: usize, k: usize) -> Vec<(usize, usize, f64)> {
        let target = ((k + 1) as f64).log2() - 1.0;
        let mut directed: HashMap<(usize, usize), f64> = HashMap::with_capacity(n * k);
        for (i, neighbours) in knn.iter().enumerate() {
            let rho = neighbours.iter().map(|&(_, dist)| dist).find(|&dist| dist > 0.0).unwrap_or(0.0);
            let mean_distance = neighbours.iter().map(|&(_, dist)| dist).sum::<f64>() / neighbours.len().max(1) as f64;
            let (mut lo, mut hi, mut sigma) = (0.0, f64::INFINITY, 1.0);
            for _ in 0..64 {
                let total: f64 = neighbours
                    .iter()
                    .map(|&(_, dist)| {
                        let gap = dist - rho;
                        if gap > 0.0 { (-gap / sigma).exp() } else { 1.0 }
                    })
                    .sum();
                if (total - target).abs() < 1e-5 {
                    break;
                }
                if total > target {
                    hi = sigma;
                    sigma = (lo + hi) / 2.0;
                } else {
                    lo = sigma;
                    sigma = if hi.is_infinite() { sigma * 2.0 } else { (lo + hi) / 2.0 };
                }
            }
            sigma = sigma.max(1e-3 * mean_distance).max(1e-12);
            for &(j, dist) in neighbours {
                let weight = if dist - rho <= 0.0 { 1.0 } else { (-(dist - rho) / sigma).exp() };
                directed.insert((i, j), weight);
            }
        }
        let mut edges = Vec::with_capacity(directed.len() * 2);
        for (&(i, j), &w) in &directed {
            match directed.get(&(j, i)) {
                Some(&v) => edges.push((i, j, w + v - w * v)),
                None => {
                    edges.push((i, j, w));
                    edges.push((j, i, w));
                }
            }
        }
        edges.sort_by_key(|e| (e.0, e.1));
        edges
    }

    pub fn fit_ab(min_dist: f64, spread: f64) -> (f64, f64) {
        let xs: Vec<f64> = (0..300).map(|i| 3.0 * spread * i as f64 / 299.0).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| if x < min_dist { 1.0 } else { (-(x - min_dist) / spread).exp() }).collect();
        let error = |a: f64, b: f64| -> f64 {
            xs.iter().zip(&ys).map(|(&x, &y)| (1.0 / (1.0 + a * x.powf(2.0 * b)) - y).powi(2)).sum()
        };
        let (mut a, mut b) = (1.0, 1.0);
        let (mut a_step, mut b_step) = (1.0, 0.5);
        for _ in 0..60 {
            let mut best = (error(a, b), a, b);
            for da in [-1.0, 0.0, 1.0] {
                for db in [-1.0, 0.0, 1.0] {
                    let (ca, cb) = (a + da * a_step, b + db * b_step);
                    if ca <= 0.0 || cb <= 0.0 {
                        continue;
                    }
                    let e = error(ca, cb);
                    if e < best.0 {
                        best = (e, ca, cb);
                    }
                }
            }
            if best.1 == a && best.2 == b {
                a_step /= 2.0;
                b_step /= 2.0;
            }
            (a, b) = (best.1, best.2);
        }
        (a, b)
    }

    /// The first principal directions, scaled to [-10, 10] per axis as
    /// umap-learn scales its start, with a little jitter.
    fn initial(data: &[f32], n: usize, d: usize, dims: usize) -> Vec<f64> {
        let components = super::pca(data, n, d, dims, &Progress::default()).unwrap_or_default();
        let m = super::mean(data, n, d);
        let mut out = vec![0f64; n * dims];
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            for (c, w) in components.iter().enumerate() {
                out[i * dims + c] = row.iter().zip(w).zip(&m).map(|((x, w), m)| ((x - m) * w) as f64).sum();
            }
        }
        let mut rng = Rng::new(SEED ^ 0x9e37_79b9);
        for axis in 0..dims {
            let (lo, hi) = (0..n).map(|i| out[i * dims + axis]).fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
            let span = (hi - lo).max(1e-12);
            for i in 0..n {
                let v = &mut out[i * dims + axis];
                *v = 20.0 * (*v - lo) / span - 10.0 + 1e-4 * (rng.unit() - 0.5);
            }
        }
        out
    }

    fn clip(v: f64) -> f64 {
        v.clamp(-4.0, 4.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn optimise(
        embedding: &mut [f64],
        edges: &[(usize, usize, f64)],
        n: usize,
        dims: usize,
        a: f64,
        b: f64,
        epochs: usize,
        progress: &Progress,
    ) -> Option<()> {
        let max_weight = edges.iter().map(|e| e.2).fold(0.0, f64::max);
        let edges: Vec<&(usize, usize, f64)> = edges.iter().filter(|e| e.2 >= max_weight / epochs as f64).collect();
        let per_sample: Vec<f64> = edges.iter().map(|e| max_weight / e.2).collect();
        let per_negative: Vec<f64> = per_sample.iter().map(|s| s / NEGATIVE_SAMPLE_RATE as f64).collect();
        let mut next_sample = per_sample.clone();
        let mut next_negative = per_negative.clone();
        let mut rng = Rng::new(SEED);
        let mut delta = [0f64; 3];

        for epoch in 0..epochs {
            if progress.stopped() {
                return None;
            }
            progress.at(0.5 + 0.5 * epoch as f64 / epochs as f64);
            let alpha = 1.0 - epoch as f64 / epochs as f64;
            let now = epoch as f64;
            for (e, &&(i, j, _)) in edges.iter().enumerate() {
                if next_sample[e] > now {
                    continue;
                }
                let mut dist_sq = 0.0;
                for c in 0..dims {
                    delta[c] = embedding[i * dims + c] - embedding[j * dims + c];
                    dist_sq += delta[c] * delta[c];
                }
                if dist_sq > 0.0 {
                    let coefficient = -2.0 * a * b * dist_sq.powf(b - 1.0) / (a * dist_sq.powf(b) + 1.0);
                    for c in 0..dims {
                        let g = clip(coefficient * delta[c]) * alpha;
                        embedding[i * dims + c] += g;
                        embedding[j * dims + c] -= g;
                    }
                }
                next_sample[e] += per_sample[e];

                let negatives = ((now - next_negative[e]) / per_negative[e]).floor().max(0.0) as usize;
                for _ in 0..negatives {
                    let other = rng.below(n);
                    if other == i {
                        continue;
                    }
                    let mut dist_sq = 0.0;
                    for c in 0..dims {
                        delta[c] = embedding[i * dims + c] - embedding[other * dims + c];
                        dist_sq += delta[c] * delta[c];
                    }
                    for c in 0..dims {
                        let g = if dist_sq > 0.0 {
                            clip(2.0 * b / ((0.001 + dist_sq) * (a * dist_sq.powf(b) + 1.0)) * delta[c])
                        } else {
                            4.0
                        };
                        embedding[i * dims + c] += g * alpha;
                    }
                }
                next_negative[e] += negatives as f64 * per_negative[e];
            }
        }
        Some(())
    }

    /// xorshift64*, as the server's.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stelly_core::vectors::{VectorBlock, VectorsHead, quantise};

    /// Three tight groups far apart in a 4-column space.
    fn groups() -> Vectors {
        let centres = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let n = 300;
        let mut v = Vectors {
            head: VectorsHead {
                n,
                dims: 4,
                blocks: vec![VectorBlock { name: "all".into(), start: 0, end: 4, columns: vec!["a".into(), "b".into(), "c".into(), "d".into()], default_weight: 1.0 }],
                ..Default::default()
            },
            ..Default::default()
        };
        for i in 0..n {
            let c = centres[i % 3];
            let wobble = |k: usize| ((i * 7 + k * 13) % 11) as f32 / 110.0;
            v.ids.push(i as i64);
            v.xy.push([i as f32, 0.0]);
            v.rows.extend((0..4).map(|k| quantise(c[k] + wobble(k))));
        }
        v
    }

    /// How often a track's nearest neighbour in the picture is from its own
    /// group: what a projection of this space has to keep.
    fn purity(points: &[[f32; 3]]) -> f32 {
        let dist = |a: &[f32; 3], b: &[f32; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>();
        let same = (0..points.len())
            .filter(|&i| {
                let nearest = (0..points.len()).filter(|&j| j != i).min_by(|&a, &b| dist(&points[i], &points[a]).total_cmp(&dist(&points[i], &points[b]))).unwrap();
                nearest % 3 == i % 3
            })
            .count();
        same as f32 / points.len() as f32
    }

    #[test]
    fn pca_and_umap_keep_groups_apart() {
        let v = groups();
        let unit = weighted(&v, &[1.0]);
        for projection in [Projection::Pca { dims: 2 }, Projection::Pca { dims: 3 }, Projection::Umap { dims: 2, sample: 1000, neighbours: 10 }, Projection::Umap { dims: 3, sample: 1000, neighbours: 10 }] {
            let points = run(&projection, &v, &unit, &Progress::default()).unwrap();
            assert_eq!(points.len(), v.head.n);
            assert!(points.iter().all(|p| p.iter().all(|x| x.is_finite())), "{projection:?}");
            let purity = purity(&points);
            assert!(purity >= 0.95, "{projection:?}: {purity}");
            if projection.dims() == 2 {
                assert!(points.iter().all(|p| p[2] == 0.0));
            }
        }
    }

    #[test]
    fn a_sample_leaves_the_rest_out_and_is_stable() {
        let v = groups();
        let unit = weighted(&v, &[1.0]);
        let projection = Projection::Umap { dims: 2, sample: 90, neighbours: 8 };
        let points = run(&projection, &v, &unit, &Progress::default()).unwrap();
        assert_eq!(points.iter().filter(|p| p[0].is_finite()).count(), 90);
        assert_eq!(sample_rows(&v, 90), sample_rows(&v, 90));
        let small: std::collections::HashSet<usize> = sample_rows(&v, 60).into_iter().collect();
        assert!(small.iter().all(|i| sample_rows(&v, 90).contains(i)));
    }

    #[test]
    fn axes_by_rank_span_the_box() {
        let v = groups();
        let unit = weighted(&v, &[1.0]);
        let points = run(&Projection::Axes { columns: [Some(0), Some(3), None], spread: true }, &v, &unit, &Progress::default()).unwrap();
        let xs: Vec<f32> = points.iter().map(|p| p[0]).collect();
        let lo = xs.iter().copied().fold(f32::MAX, f32::min);
        let hi = xs.iter().copied().fold(f32::MIN, f32::max);
        assert!(lo < -0.9 && hi > 0.9, "{lo} {hi}");
    }

    #[test]
    fn a_cancelled_umap_returns_nothing() {
        let v = groups();
        let unit = weighted(&v, &[1.0]);
        let progress = Progress::default();
        progress.cancelled.store(true, Ordering::Relaxed);
        assert!(run(&Projection::Umap { dims: 2, sample: 1000, neighbours: 10 }, &v, &unit, &progress).is_none());
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;
    use stelly_core::vectors::{VectorBlock, VectorsHead, quantise};

    /// The production catalogue's size, in clusters: how long each step
    /// takes there. `cargo test --release bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "a benchmark"]
    fn at_catalogue_size() {
        let (n, d, clusters) = (156_000, 81, 60);
        let mut seed = 1u64;
        let mut unit = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        let centres: Vec<Vec<f32>> = (0..clusters).map(|_| (0..d).map(|_| unit()).collect()).collect();
        let mut v = Vectors {
            head: VectorsHead { n, dims: d, blocks: vec![VectorBlock { name: "all".into(), start: 0, end: d, columns: (0..d).map(|c| c.to_string()).collect(), default_weight: 1.0 }], genres: (0..100).map(|g| g.to_string()).collect(), ..Default::default() },
            ..Default::default()
        };
        for i in 0..n {
            let c = &centres[i % clusters];
            v.ids.push(i as i64);
            v.genre.push((i % clusters) as u16);
            v.rows.extend(c.iter().map(|x| quantise(x + unit() * 0.3)));
        }
        let t = Instant::now();
        let unit = weighted(&v, &[1.0]);
        println!("weighted: {:?}", t.elapsed());
        for p in [Projection::Pca { dims: 3 }, Projection::Axes { columns: [Some(0), Some(1), Some(2)], spread: true }, Projection::Umap { dims: 3, sample: 25_000, neighbours: 15 }] {
            let t = Instant::now();
            let points = run(&p, &v, &unit, &Progress::default()).unwrap();
            let took = t.elapsed();
            let t = Instant::now();
            let regions = crate::labels::find(&points, &v.genre, 100, p.dims(), |_| true);
            println!("{p:?}: {took:?}, labels {:?} ({} regions)", t.elapsed(), regions.len());
        }
    }
}
