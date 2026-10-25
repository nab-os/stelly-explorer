//! Each mix-analysed track as the path its phrases take through the space:
//! an intro in one place, the drop in another, the outro back again.
//!
//! A phrase has only its sound of its own, so it is its track's row with
//! the semantic block put in from the phrase, weighted and brought to unit
//! length as a track's is. Where to draw it is where the tracks most like it
//! are drawn: the mean of its nearest tracks' places, weighted by how near,
//! which holds in any projection, the server's map and UMAP included, that
//! cannot place a new point by themselves. Its nearest are looked for among
//! its own track's nearest few hundred, where a phrase of it lands; over the
//! whole space it would be hours.

use crate::project::Progress;
use crate::render::Line;
use std::collections::HashMap;
use std::ops::Range;
use stelly_core::phrases::Phrases;
use stelly_core::vectors::{Vectors, unquantise};

/// Tracks a phrase's neighbours are looked for among: its track's nearest.
const CANDIDATES: usize = 256;
/// Neighbours a phrase is placed by.
pub const NEAR: usize = 8;
/// How sharply the nearest count over the rest, in cosine: within a few
/// hundredths of the nearest is most of the weight.
const SHARPNESS: f32 = 0.01;

/// A phrase's nearest tracks, as `Placement::near` holds them, by phrase.
type Placed = Vec<(usize, [(u32, f32); NEAR])>;

pub struct Placement {
    /// Per track with phrases: its row, and its phrases' range.
    pub tracks: Vec<(usize, Range<usize>)>,
    /// Seconds into its track each phrase starts.
    pub starts: Vec<f32>,
    /// Per phrase, its nearest tracks' rows and their weights, summing to 1.
    pub near: Vec<[(u32, f32); NEAR]>,
}

/// Where each phrase sits among the tracks, by `unit`, the tracks' weighted
/// rows, and `weights`, the blocks' weights that made them. `None` when cancelled.
pub fn place(v: &Vectors, unit: &[f32], weights: &[f32], phrases: &Phrases, row_of: &HashMap<i64, usize>, progress: &Progress) -> Option<Placement> {
    let d = v.head.dims;
    let n = v.head.n;
    let mut scale = vec![1.0f32; d];
    for (b, w) in v.head.blocks.iter().zip(weights) {
        scale[b.start..b.end].fill(*w);
    }
    let (start, dims) = (phrases.head.start, phrases.head.dims);
    let offsets = phrases.offsets();
    let tracks: Vec<(usize, Range<usize>)> = phrases
        .ids
        .iter()
        .enumerate()
        .filter_map(|(t, id)| Some((*row_of.get(id)?, offsets[t]..offsets[t + 1])))
        .filter(|_| start + dims <= d)
        .collect();

    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(32);
    let chunk = tracks.len().div_ceil(threads).max(1);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = tracks.len().max(1);
    let parts: Vec<Placed> = std::thread::scope(|scope| {
        let handles: Vec<_> = tracks
            .chunks(chunk)
            .map(|part| {
                let (scale, done) = (&scale, &done);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    let mut sims = vec![0f32; n];
                    let mut row = vec![0f32; d];
                    for (r, range) in part {
                        if progress.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                            return out;
                        }
                        // The track's own nearest, among which its phrases' are.
                        let mine = &unit[r * d..(r + 1) * d];
                        for (j, s) in sims.iter_mut().enumerate() {
                            *s = mine.iter().zip(&unit[j * d..(j + 1) * d]).map(|(a, b)| a * b).sum();
                        }
                        let mut candidates: Vec<u32> = (0..n as u32).collect();
                        let c = CANDIDATES.min(n);
                        if c < n {
                            candidates.select_nth_unstable_by(c - 1, |a, b| sims[*b as usize].total_cmp(&sims[*a as usize]));
                            candidates.truncate(c);
                        }
                        for phrase in range.clone() {
                            // The track's row, the phrase's sound in it.
                            for (k, x) in row.iter_mut().enumerate() {
                                *x = unquantise(v.row(*r)[k]);
                            }
                            for (k, q) in phrases.row(phrase).iter().enumerate() {
                                row[start + k] = unquantise(*q);
                            }
                            let mut norm = 0f32;
                            for (x, s) in row.iter_mut().zip(scale) {
                                *x *= s;
                                norm += *x * *x;
                            }
                            let norm = norm.sqrt().max(1e-9);
                            let mut scored: Vec<(u32, f32)> = candidates
                                .iter()
                                .map(|&j| (j, row.iter().zip(&unit[j as usize * d..(j as usize + 1) * d]).map(|(a, b)| a * b).sum::<f32>() / norm))
                                .collect();
                            let k = NEAR.min(scored.len());
                            scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
                            scored.truncate(k);
                            let best = scored.iter().map(|s| s.1).fold(f32::MIN, f32::max);
                            let mut near = [(0u32, 0f32); NEAR];
                            let mut total = 0f32;
                            for (slot, (j, sim)) in near.iter_mut().zip(scored) {
                                let w = ((sim - best) / SHARPNESS).exp();
                                *slot = (j, w);
                                total += w;
                            }
                            near.iter_mut().for_each(|slot| slot.1 /= total.max(1e-12));
                            out.push((phrase, near));
                        }
                        let finished = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if finished % 64 == 0 {
                            progress.done.store((finished * 1000 / total) as u32, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    if progress.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let mut near = vec![[(0u32, 0f32); NEAR]; phrases.starts.len()];
    for (phrase, slot) in parts.into_iter().flatten() {
        near[phrase] = slot;
    }
    progress.done.store(1000, std::sync::atomic::Ordering::Relaxed);
    Some(Placement { tracks, starts: phrases.starts.clone(), near })
}

impl Placement {
    /// Each phrase's place in a projection, from its neighbours' places;
    /// NaN where none of them is placed, as outside a UMAP's sample.
    pub fn positions(&self, positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
        self.near
            .iter()
            .map(|near| {
                let mut p = [0f32; 3];
                let mut total = 0f32;
                for &(j, w) in near {
                    let q = positions[j as usize];
                    if w > 0.0 && q[0].is_finite() {
                        (0..3).for_each(|a| p[a] += q[a] * w);
                        total += w;
                    }
                }
                if total > 0.0 { p.map(|x| x / total) } else { [f32::NAN; 3] }
            })
            .collect()
    }

    /// The paths as lines, a smooth curve through each track's phrases in
    /// order, for the tracks `drawn` lets through, in their own colours.
    pub fn lines(&self, phrase_positions: &[[f32; 3]], colour: impl Fn(usize) -> Option<[u8; 4]>) -> Vec<Line> {
        let mut out = Vec::new();
        for (row, range) in &self.tracks {
            let Some(colour) = colour(*row) else { continue };
            let points: Vec<[f32; 3]> = phrase_positions[range.clone()].iter().copied().filter(|p| p[0].is_finite()).collect();
            let curve = smooth(&points, 3);
            out.extend(curve.windows(2).map(|w| Line { from: w[0], to: w[1], colour }));
        }
        out
    }
}

/// A Catmull-Rom curve through `points`, `steps` pieces between each two:
/// through every phrase, without the corners a polyline has.
pub fn smooth(points: &[[f32; 3]], steps: usize) -> Vec<[f32; 3]> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let at = |i: isize| points[i.clamp(0, points.len() as isize - 1) as usize];
    let mut out = Vec::with_capacity((points.len() - 1) * steps + 1);
    for i in 0..points.len() as isize - 1 {
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        for s in 0..steps {
            let t = s as f32 / steps as f32;
            let (t2, t3) = (t * t, t * t * t);
            out.push(std::array::from_fn(|a| {
                0.5 * (2.0 * p1[a] + (-p0[a] + p2[a]) * t + (2.0 * p0[a] - 5.0 * p1[a] + 4.0 * p2[a] - p3[a]) * t2 + (-p0[a] + 3.0 * p1[a] - 3.0 * p2[a] + p3[a]) * t3)
            }));
        }
    }
    out.push(points[points.len() - 1]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project;
    use stelly_core::phrases::PhrasesHead;
    use stelly_core::vectors::{VectorBlock, VectorsHead, quantise};

    /// Two groups of tracks, one sounding one way and placed left, the other
    /// sounding another and placed right; tempo alike throughout.
    fn two_groups() -> Vectors {
        let n = 200;
        let mut v = Vectors {
            head: VectorsHead {
                n,
                dims: 3,
                blocks: vec![
                    VectorBlock { name: "tempo".into(), start: 0, end: 1, columns: vec!["t".into()], default_weight: 1.0 },
                    VectorBlock { name: "semantic".into(), start: 1, end: 3, columns: vec!["a".into(), "b".into()], default_weight: 2.0 },
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        for i in 0..n {
            let wobble = (i % 7) as f32 / 100.0;
            let sound = if i < n / 2 { [1.0 - wobble, wobble] } else { [wobble, 1.0 - wobble] };
            v.ids.push(i as i64);
            v.xy.push([if i < n / 2 { -1.0 } else { 1.0 }, 0.0]);
            v.rows.extend([quantise(0.5), quantise(sound[0]), quantise(sound[1])]);
        }
        v
    }

    #[test]
    fn a_phrase_lands_among_the_tracks_it_sounds_like() {
        let v = two_groups();
        let weights = [1.0, 2.0];
        let unit = project::weighted(&v, &weights);
        let row_of: HashMap<i64, usize> = v.ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        // Track 0, on the left: a first phrase like its own group, a second
        // like the other.
        let phrases = Phrases {
            head: PhrasesHead { block: "semantic".into(), start: 1, dims: 2, tracks: 1, phrases: 2, ..Default::default() },
            ids: vec![0],
            counts: vec![2],
            starts: vec![8.0, 24.0],
            rows: vec![quantise(1.0), quantise(0.0), quantise(0.0), quantise(1.0)],
        };
        let placement = place(&v, &unit, &weights, &phrases, &row_of, &Progress::default()).unwrap();
        let positions: Vec<[f32; 3]> = v.xy.iter().map(|p| [p[0], p[1], 0.0]).collect();
        let at = placement.positions(&positions);
        assert!(at[0][0] < -0.9, "the phrase like its own group stays with it: {:?}", at[0]);
        assert!(at[1][0] > 0.9, "the phrase like the other group goes to it: {:?}", at[1]);
        for near in &placement.near {
            assert!((near.iter().map(|s| s.1).sum::<f32>() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn the_curve_goes_through_every_phrase() {
        let points = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 0.0], [4.0, 4.0, 0.0]];
        let curve = smooth(&points, 3);
        assert_eq!(curve.len(), 3 * 3 + 1);
        for (k, p) in points.iter().enumerate() {
            let q = curve[k * 3];
            assert!((0..3).all(|a| (q[a] - p[a]).abs() < 1e-5), "{k}: {q:?} is not {p:?}");
        }
    }
}

#[cfg(test)]
mod against {
    use super::*;
    use crate::project;
    use std::time::Instant;
    use stelly_core::phrases::PhrasesHead;
    use stelly_core::vectors::{VectorBlock, VectorsHead, quantise};

    /// From a real server, `STELLY_SERVER` and `STELLY_TOKEN`: the space and
    /// its phrases, placed and drawn.
    #[test]
    #[ignore = "needs a Stelly server with /api/mix/phrases"]
    fn places_a_server_s_phrases() {
        let server = crate::source::Server::find().expect("a server");
        let (v, _) = crate::source::load(&server).unwrap();
        let (phrases, _) = crate::source::phrases(&server).unwrap();
        let weights: Vec<f32> = v.head.blocks.iter().map(|b| b.default_weight).collect();
        let unit = project::weighted(&v, &weights);
        let row_of: HashMap<i64, usize> = v.ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let placement = place(&v, &unit, &weights, &phrases, &row_of, &Progress::default()).unwrap();
        let map = project::run(&project::Projection::Map, &v, &unit, &Progress::default()).unwrap();
        let at = placement.positions(&map);
        assert_eq!(at.len(), phrases.head.phrases);
        assert!(at.iter().all(|p| p[0].is_finite()), "every phrase has a place on the map");
        let lines = placement.lines(&at, |_| Some([255; 4]));
        println!("{} tracks, {} phrases, {} line pieces", placement.tracks.len(), at.len(), lines.len());
        assert!(!lines.is_empty());
    }

    /// The production library's size: 156,000 tracks, 19,000 of them with
    /// 20 phrases. `cargo test --release at_library_size -- --ignored --nocapture`.
    #[test]
    #[ignore = "a benchmark"]
    fn at_library_size() {
        let (n, d, with_phrases, each) = (156_000, 81, 19_000, 20);
        let mut seed = 3u64;
        let mut unit_random = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        let mut v = Vectors {
            head: VectorsHead {
                n,
                dims: d,
                blocks: vec![
                    VectorBlock { name: "rest".into(), start: 0, end: 41, columns: (0..41).map(|c| c.to_string()).collect(), default_weight: 1.0 },
                    VectorBlock { name: "semantic".into(), start: 41, end: 81, columns: (0..40).map(|c| c.to_string()).collect(), default_weight: 2.0 },
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        for i in 0..n {
            v.ids.push(i as i64);
            v.rows.extend((0..d).map(|_| quantise(unit_random())));
        }
        let mut phrases = Phrases { head: PhrasesHead { block: "semantic".into(), start: 41, dims: 40, ..Default::default() }, ..Default::default() };
        for t in 0..with_phrases {
            phrases.ids.push((t * (n / with_phrases)) as i64);
            phrases.counts.push(each as u16);
            for k in 0..each {
                phrases.starts.push(k as f32 * 16.0);
                phrases.rows.extend((0..40).map(|_| quantise(unit_random())));
            }
        }
        phrases.head.tracks = with_phrases;
        phrases.head.phrases = with_phrases * each;
        let weights = [1.0, 2.0];
        let unit = project::weighted(&v, &weights);
        let row_of: HashMap<i64, usize> = v.ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let t = Instant::now();
        let placement = place(&v, &unit, &weights, &phrases, &row_of, &Progress::default()).unwrap();
        println!("placing {} phrases: {:?}", placement.near.len(), t.elapsed());
        let positions = vec![[0.5f32, 0.5, 0.0]; n];
        let t = Instant::now();
        let at = placement.positions(&positions);
        println!("their places in a projection: {:?}", t.elapsed());
        let t = Instant::now();
        let lines = placement.lines(&at, |_| Some([255; 4]));
        println!("{} line pieces: {:?}", lines.len(), t.elapsed());
    }
}
