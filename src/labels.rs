//! Where each genre gathers, named over the points: the Stelly app's map
//! labels (`app/assets/map.js`, `findRegions`), in any projection and in 3D.
//!
//! The projection is cut into a coarse grid, a square one in 2D and a cube in
//! 3D; each genre's counts are summed over each cell's neighbourhood, and a
//! cell that is a local peak of that, held mostly by the genre and clearly
//! denser than the genre is on average, is a region. The name sits on the
//! mean of the genre's points around the peak. Placing them on screen, the
//! strongest first and none over another, is the app's, per frame.

/// A place a genre gathers, in projection space.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub position: [f32; 3],
    pub genre: u16,
    pub weight: f32,
}

pub fn find(positions: &[[f32; 3]], genre: &[u16], genres: usize, dims: usize, named: impl Fn(u16) -> bool) -> Vec<Region> {
    let n = positions.len();
    if n == 0 || genres == 0 {
        return Vec::new();
    }
    // Cells per side: fewer in 3D, where each has 27 neighbours' worth.
    let side: usize = if dims == 3 { 18 } else { 40 };
    let least = (n as f32 / 2000.0).max(10.0);
    let share = 0.4;
    let apart = if dims == 3 { 3 } else { 5 };
    let (first, more) = (1.5, 3.0);

    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in positions.iter().filter(|p| p[0].is_finite()) {
        for a in 0..dims {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    if lo[0] > hi[0] {
        return Vec::new();
    }
    let cell_of = |p: &[f32; 3]| -> [usize; 3] {
        let mut c = [0usize; 3];
        for a in 0..dims {
            let t = (p[a] - lo[a]) / (hi[a] - lo[a]).max(1e-9) * 0.9999;
            c[a] = ((t * side as f32) as usize).min(side - 1);
        }
        c
    };
    let cells = side.pow(dims as u32);
    let index = |c: [usize; 3]| c[0] + side * (c[1] + side * c[2]);

    let mut count = vec![0f32; cells * genres];
    let mut total = vec![0f32; cells];
    let mut size = vec![0f32; genres];
    for (p, g) in positions.iter().zip(genre) {
        if !p[0].is_finite() {
            continue;
        }
        let c = index(cell_of(p));
        count[*g as usize * cells + c] += 1.0;
        total[c] += 1.0;
        size[*g as usize] += 1.0;
    }
    let occupied = total.iter().filter(|t| **t > 0.0).count().max(1) as f32;
    let window = 3f32.powi(dims as i32);

    // Sums over each cell's 3 (or 3x3, or 3x3x3) neighbourhood, one axis at a time.
    let neighbourhood = |values: &[f32]| -> Vec<f32> {
        let mut out = values.to_vec();
        for axis in 0..dims {
            let stride = side.pow(axis as u32);
            let mut next = vec![0f32; cells];
            for c in 0..cells {
                let at = (c / stride) % side;
                let mut sum = out[c];
                if at > 0 {
                    sum += out[c - stride];
                }
                if at + 1 < side {
                    sum += out[c + stride];
                }
                next[c] = sum;
            }
            out = next;
        }
        out
    };
    let around = neighbourhood(&total);

    let coords = |c: usize| -> [usize; 3] { [c % side, (c / side) % side, if dims == 3 { c / (side * side) } else { 0 }] };
    let mut peaks: Vec<(u16, [usize; 3], f32)> = Vec::new();
    for g in 0..genres {
        if size[g] < least || !named(g as u16) {
            continue;
        }
        let mine = neighbourhood(&count[g * cells..(g + 1) * cells]);
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for c in 0..cells {
            let v = mine[c];
            if v < least || v < around[c] * share {
                continue;
            }
            // A local peak within two cells; ties to the lower cell.
            let at = coords(c);
            let mut top = true;
            'search: for dz in if dims == 3 { -2i32..=2 } else { 0..=0 } {
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let q = [at[0] as i32 + dx, at[1] as i32 + dy, at[2] as i32 + dz];
                        if q.iter().any(|x| *x < 0 || *x >= side as i32) {
                            continue;
                        }
                        let k = index([q[0] as usize, q[1] as usize, q[2] as usize]);
                        if mine[k] > v || (mine[k] == v && k < c) {
                            top = false;
                            break 'search;
                        }
                    }
                }
            }
            if top {
                candidates.push((c, v));
            }
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let even = window * size[g] / occupied;
        let mut kept: Vec<[usize; 3]> = Vec::new();
        for (c, v) in candidates {
            if v < even * if kept.is_empty() { first } else { more } {
                continue;
            }
            let at = coords(c);
            let far = kept.iter().all(|q| (0..3).map(|a| at[a].abs_diff(q[a])).max().unwrap_or(0) >= apart);
            if far {
                kept.push(at);
                peaks.push((g as u16, at, v));
            }
        }
    }

    // The name on the mean of the genre's own points around each peak.
    let mut sums = vec![([0f64; 3], 0usize); peaks.len()];
    for (p, g) in positions.iter().zip(genre) {
        if !p[0].is_finite() {
            continue;
        }
        let at = cell_of(p);
        for (k, (pg, pc, _)) in peaks.iter().enumerate() {
            if pg == g && (0..dims).all(|a| at[a].abs_diff(pc[a]) <= 1) {
                (0..3).for_each(|a| sums[k].0[a] += p[a] as f64);
                sums[k].1 += 1;
            }
        }
    }
    let mut regions: Vec<Region> = peaks
        .iter()
        .zip(sums)
        .filter(|(_, (_, m))| *m > 0)
        .map(|((g, _, w), (s, m))| Region { position: [(s[0] / m as f64) as f32, (s[1] / m as f64) as f32, (s[2] / m as f64) as f32], genre: *g, weight: *w })
        .collect();
    regions.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(dims: usize) -> (Vec<[f32; 3]>, Vec<u16>) {
        let mut positions = Vec::new();
        let mut genre = Vec::new();
        let mut seed = 7u64;
        let mut unit = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        // Genre 0 in two islands, 1 in one, 2 evenly everywhere, 3 too few.
        let islands: [(u16, [f32; 3], usize); 4] = [(0, [-5.0, -5.0, -5.0], 3000), (0, [6.0, 5.0, 4.0], 1200), (1, [4.0, -4.0, 3.0], 2500), (3, [0.0, 0.0, 0.0], 8)];
        for (g, c, k) in islands {
            for _ in 0..k {
                let mut p = [0f32; 3];
                for a in 0..dims {
                    p[a] = c[a] + (unit() + unit() + unit() - 1.5) * 1.6;
                }
                positions.push(p);
                genre.push(g);
            }
        }
        for _ in 0..1500 {
            let mut p = [0f32; 3];
            for x in p.iter_mut().take(dims) {
                *x = unit() * 20.0 - 10.0;
            }
            positions.push(p);
            genre.push(2);
        }
        (positions, genre)
    }

    #[test]
    fn islands_are_named_and_a_spread_genre_is_not() {
        for dims in [2, 3] {
            let (positions, genre) = cloud(dims);
            let regions = find(&positions, &genre, 4, dims, |_| true);
            let names: Vec<u16> = regions.iter().map(|r| r.genre).collect();
            assert_eq!(names.iter().filter(|g| **g == 0).count(), 2, "{dims}D: {regions:?}");
            assert_eq!(names.iter().filter(|g| **g == 1).count(), 1, "{dims}D");
            assert!(!names.contains(&2) && !names.contains(&3), "{dims}D: {names:?}");
            let near = |r: &Region, c: [f32; 2]| (r.position[0] - c[0]).abs() < 1.0 && (r.position[1] - c[1]).abs() < 1.0;
            assert!(regions.iter().any(|r| r.genre == 1 && near(r, [4.0, -4.0])), "{dims}D");
        }
    }

    #[test]
    fn unnamed_genres_are_skipped() {
        let (positions, genre) = cloud(2);
        assert!(find(&positions, &genre, 4, 2, |g| g != 0).iter().all(|r| r.genre != 0));
    }
}
