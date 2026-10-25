//! What a point's colour says. Each mode gives one RGBA per track and the
//! legend that reads it back; dimming for a search or a focus is applied on
//! top, by the app.
//!
//! The palettes follow the dataviz rules: categorical hues in a fixed,
//! colourblind-checked order and never cycled, so genres past the eighth
//! fold into "other"; one hue for a magnitude; two hues and a grey middle
//! for a signed value. The key wheel is the exception a wheel has to be, and
//! the legend names every key so the colour is never alone.

use crate::mix::Readiness;
use stelly_core::vectors::{Key, Vectors, unquantise};

pub type Rgba = [u8; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Colouring {
    Genre,
    Key,
    Tempo,
    Column(usize),
    /// Whether stelly-mix has analysed it whole, and whether a set can use it.
    Mix,
}

pub enum Legend {
    /// (name, colour, tracks), biggest first; "other" last.
    Categories(Vec<(String, Rgba, usize)>),
    /// A ramp from `low` to `high`, labelled at its ends and middle.
    Ramp { stops: Vec<Rgba>, low: String, middle: String, high: String },
    /// The Camelot wheel: for each of 1 to 12, the major (B) and minor (A).
    Wheel { major: [Rgba; 12], minor: [Rgba; 12], unsure: usize },
    /// Where tracks stand for mixing: (name, colour, tracks, which).
    Mix(Vec<(String, Rgba, usize, Readiness)>),
}

/// The validated categorical slots, stepped for a dark surface.
pub const CATEGORICAL: [Rgba; 8] = [
    rgb(0x39, 0x87, 0xe5),
    rgb(0xd9, 0x59, 0x26),
    rgb(0x19, 0x9e, 0x70),
    rgb(0xc9, 0x85, 0x00),
    rgb(0xd5, 0x51, 0x81),
    rgb(0x00, 0x83, 0x00),
    rgb(0x90, 0x85, 0xe9),
    rgb(0xe6, 0x67, 0x67),
];

pub const MUTED: Rgba = rgb(0x6b, 0x70, 0x82);

/// Below this the analyser was guessing, and the key is drawn as unsure.
pub const KEY_SURE: u8 = 56;

const TEMPO_LOW: f32 = 60.0;
const TEMPO_HIGH: f32 = 180.0;

const fn rgb(r: u8, g: u8, b: u8) -> Rgba {
    [r, g, b, 255]
}

/// `readiness` is each track's, by row; empty while the mix library has not
/// come, when every track is drawn as not analysed.
pub fn colour(v: &Vectors, mode: Colouring, readiness: &[Readiness]) -> (Vec<Rgba>, Legend) {
    match mode {
        Colouring::Genre => genre(v),
        Colouring::Key => key(v),
        Colouring::Tempo => tempo(v),
        Colouring::Column(c) => column(v, c),
        Colouring::Mix => mix(v, readiness),
    }
}

/// Ready in the green slot, drifting in the amber one, the rest grey: a
/// state, so the legend names each and the colour is never alone.
pub const READY: Rgba = CATEGORICAL[2];
pub const DRIFTING: Rgba = CATEGORICAL[3];

fn mix(v: &Vectors, readiness: &[Readiness]) -> (Vec<Rgba>, Legend) {
    let of = |i: usize| readiness.get(i).copied().unwrap_or(Readiness::NotAnalysed);
    let colour = |r: Readiness| match r {
        Readiness::Ready => READY,
        Readiness::Drifting => DRIFTING,
        Readiness::NotAnalysed => MUTED,
    };
    let colours = (0..v.head.n).map(|i| colour(of(i))).collect();
    let count = |r: Readiness| (0..v.head.n).filter(|&i| of(i) == r).count();
    let legend = [
        ("Ready to mix", Readiness::Ready),
        ("Analysed, tempo drifts", Readiness::Drifting),
        ("Not analysed for mixing", Readiness::NotAnalysed),
    ]
    .into_iter()
    .map(|(name, r)| (name.to_string(), colour(r), count(r), r))
    .collect();
    (colours, Legend::Mix(legend))
}

fn genre(v: &Vectors) -> (Vec<Rgba>, Legend) {
    let mut counts = vec![0usize; v.head.genres.len().max(1)];
    v.genre.iter().for_each(|g| counts[*g as usize] += 1);
    let mut order: Vec<usize> = (0..counts.len()).filter(|g| counts[*g] > 0).collect();
    order.sort_by(|a, b| counts[*b].cmp(&counts[*a]).then(a.cmp(b)));
    let mut slot = vec![None; counts.len()];
    for (s, g) in order.iter().take(CATEGORICAL.len()).enumerate() {
        slot[*g] = Some(s);
    }
    let colours = v.genre.iter().map(|g| slot[*g as usize].map_or(MUTED, |s| CATEGORICAL[s])).collect();

    let name = |g: usize| v.head.genres.get(g).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| "no genre".into());
    let mut legend: Vec<(String, Rgba, usize)> = order.iter().take(CATEGORICAL.len()).enumerate().map(|(s, g)| (name(*g), CATEGORICAL[s], counts[*g])).collect();
    let rest: usize = order.iter().skip(CATEGORICAL.len()).map(|g| counts[*g]).sum();
    if rest > 0 {
        let genres = order.len() - CATEGORICAL.len();
        legend.push((format!("{genres} other genres"), MUTED, rest));
    }
    (colours, Legend::Categories(legend))
}

/// Camelot number n's colour: hue round the wheel, so neighbours that mix
/// well look alike; lighter for major.
pub fn key_colour(number: u8, minor: bool) -> Rgba {
    let hue = (number as f32 - 1.0) * 30.0 + 25.0;
    if minor { oklch(0.62, 0.15, hue) } else { oklch(0.8, 0.13, hue) }
}

fn key(v: &Vectors) -> (Vec<Rgba>, Legend) {
    let mut unsure = 0;
    let colours = (0..v.head.n)
        .map(|i| match v.key(i) {
            Some(k) if v.key_strength[i] >= KEY_SURE => {
                let (number, letter) = k.camelot();
                key_colour(number, letter == 'A')
            }
            _ => {
                unsure += 1;
                MUTED
            }
        })
        .collect();
    let major = std::array::from_fn(|i| key_colour(i as u8 + 1, false));
    let minor = std::array::from_fn(|i| key_colour(i as u8 + 1, true));
    (colours, Legend::Wheel { major, minor, unsure })
}

/// The Camelot code and name of a track's key, for the panels.
pub fn key_label(v: &Vectors, i: usize) -> String {
    match v.key(i) {
        Some(k) => {
            let (number, letter) = k.camelot();
            let sure = v.key_strength[i] as f32 / 255.0 * 100.0;
            format!("{number}{letter} · {} ({sure:.0}% sure)", k.name())
        }
        None => "none found".into(),
    }
}

pub fn camelot_name(number: u8, minor: bool) -> String {
    // The key at that place on the wheel: 8B is C major, each step a fifth.
    let pitch = ((number as u32 + 4) * 7 % 12) as u8;
    let major = Key { pitch, minor: false };
    let key = if minor { Key { pitch: (pitch + 9) % 12, minor: true } } else { major };
    key.name()
}

fn tempo_ramp(t: f32) -> Rgba {
    oklch(0.4 + 0.52 * t, 0.05 + 0.12 * t, 75.0 - 20.0 * t)
}

fn tempo(v: &Vectors) -> (Vec<Rgba>, Legend) {
    let colours = v
        .bpm
        .iter()
        .map(|b| if b.is_finite() { tempo_ramp(((b - TEMPO_LOW) / (TEMPO_HIGH - TEMPO_LOW)).clamp(0.0, 1.0)) } else { MUTED })
        .collect();
    let stops = (0..=24).map(|i| tempo_ramp(i as f32 / 24.0)).collect();
    (colours, Legend::Ramp { stops, low: format!("{TEMPO_LOW:.0} BPM"), middle: format!("{:.0}", (TEMPO_LOW + TEMPO_HIGH) / 2.0), high: format!("{TEMPO_HIGH:.0}+ BPM") })
}

/// Grey in the middle, blue below, orange above: the categorical slots' own
/// first two hues, so the scale belongs with the rest.
fn diverging(t: f32) -> Rgba {
    let t = t.clamp(-1.0, 1.0);
    let (hue, chroma) = if t < 0.0 { (255.0, 0.15) } else { (45.0, 0.16) };
    oklch(0.55 + 0.17 * (1.0 - t.abs()), chroma * t.abs() + 0.01, hue)
}

fn column(v: &Vectors, c: usize) -> (Vec<Rgba>, Legend) {
    let values: Vec<f32> = (0..v.head.n).map(|i| unquantise(v.row(i)[c])).collect();
    let mut sorted = values.clone();
    sorted.sort_by(f32::total_cmp);
    let at = |q: f32| sorted.get(((sorted.len().max(1) - 1) as f32 * q) as usize).copied().unwrap_or(0.0);
    let (low, middle, high) = (at(0.02), at(0.5), at(0.98));
    let scale = |x: f32| if x < middle { (x - middle) / (middle - low).max(1e-6) } else { (x - middle) / (high - middle).max(1e-6) };
    let colours = values.iter().map(|x| diverging(scale(*x))).collect();
    let stops = (0..=24).map(|i| diverging(i as f32 / 12.0 - 1.0)).collect();
    let (from, to) = column_ends(v, c);
    (colours, Legend::Ramp { stops, low: from, middle: "median".into(), high: to })
}

/// What each end of a column means, where the space's labels say; the mood
/// columns are contrasts between two phrases.
fn column_ends(v: &Vectors, c: usize) -> (String, String) {
    const MOODS: [(&str, &str, &str); 8] = [
        ("energy", "calm", "energetic"),
        ("valence", "sad", "happy"),
        ("aggression", "tender", "aggressive"),
        ("danceability", "no groove", "danceable"),
        ("darkness", "bright", "dark"),
        ("acoustic", "electronic", "acoustic"),
        ("vocals", "instrumental", "vocal"),
        ("complexity", "simple", "complex"),
    ];
    let name = v.head.blocks.iter().find(|b| (b.start..b.end).contains(&c)).and_then(|b| b.columns.get(c - b.start)).cloned().unwrap_or_default();
    MOODS
        .iter()
        .find(|(m, ..)| *m == name)
        .map_or(("low".into(), "high".into()), |(_, lo, hi)| (lo.to_string(), hi.to_string()))
}

/// OKLCH to sRGB, clipped into gamut.
pub fn oklch(l: f32, c: f32, h: f32) -> Rgba {
    let (a, b) = (c * h.to_radians().cos(), c * h.to_radians().sin());
    let l_ = (l + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
    let m_ = (l - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
    let s_ = (l - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
    let linear = [
        4.076_741_7 * l_ - 3.307_711_6 * m_ + 0.230_969_94 * s_,
        -1.268_438 * l_ + 2.609_757_4 * m_ - 0.341_319_38 * s_,
        -0.004_196_086_3 * l_ - 0.703_418_6 * m_ + 1.707_614_7 * s_,
    ];
    let channel = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let v = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (v * 255.0).round() as u8
    };
    [channel(linear[0]), channel(linear[1]), channel(linear[2]), 255]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camelot_names_go_round_the_wheel() {
        assert_eq!(camelot_name(8, false), "C major");
        assert_eq!(camelot_name(8, true), "A minor");
        assert_eq!(camelot_name(9, false), "G major");
        assert_eq!(camelot_name(1, true), "A♭ minor");
        // And back: every name's key sits where it was named.
        for number in 1..=12 {
            for minor in [false, true] {
                let pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
                let found = pitches
                    .iter()
                    .filter_map(|p| Key::parse(p, Some(if minor { "minor" } else { "major" })))
                    .find(|k| k.name() == camelot_name(number, minor))
                    .unwrap();
                assert_eq!(found.camelot(), (number, if minor { 'A' } else { 'B' }));
            }
        }
    }

    #[test]
    fn oklch_greys_are_grey() {
        let [r, g, b, _] = oklch(0.6, 0.0, 0.0);
        assert!(r.abs_diff(g) <= 1 && g.abs_diff(b) <= 1);
        assert_eq!(oklch(1.0, 0.0, 0.0)[..3], [255, 255, 255]);
    }
}
