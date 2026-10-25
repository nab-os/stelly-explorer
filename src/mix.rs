//! What stelly-mix knows of the tracks, through the server: which have been
//! analysed whole for mixing, their tempo and key from end to end, the best
//! ways out of one into the next, and a set planned from one.
//!
//! The shapes are stelly-mix's own (`stelly_mix::api`), read here with only
//! the fields the explorer shows, so it links no audio analysis to read them.
//! Every call blocks: the app makes them on a thread of their own.

use crate::source::Server;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use stelly_core::api::MixState;

/// One analysed track, as `/api/mix/library` lists it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Summary {
    pub id: i64,
    /// Over the whole track, from its beat grid.
    pub bpm: Option<f64>,
    /// Made on one grid from start to end: a set can play it at one rate.
    pub steady: bool,
    /// Camelot, "8A", over the whole track.
    pub key: Option<String>,
    pub seconds: f64,
    pub phrases: usize,
}

/// Where a track stands for mixing, by catalogue row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// Analysed, and its tempo holds: a set can use it.
    Ready,
    /// Analysed, but its tempo drifts: sets leave it out.
    Drifting,
    NotAnalysed,
}

/// One way from a track into another, as `/api/mix/next` gives it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Way {
    /// The next track's id.
    pub to: i64,
    pub score: f64,
    pub sound: f64,
    /// Incoming tempo over the one it plays at: 1.02 is 2% faster.
    pub ratio: f64,
    pub key_steps: Option<u32>,
    /// Incoming loudness minus outgoing, LU.
    pub energy: Option<f64>,
    /// Seconds of the playing track mixed out of, and of the next mixed into.
    pub out_at: (f64, f64),
    pub into_at: (f64, f64),
}

/// A set the server planned: its tracks in order and the way between each.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Plan {
    pub tracks: Vec<i64>,
    pub ways: Vec<Way>,
    pub seconds: f64,
    pub mean_score: f64,
}

/// `stelly_mix::api::SetRequest`, every field: the server takes no defaults.
#[derive(Serialize)]
struct SetRequest {
    first: Option<i64>,
    like: Option<String>,
    minutes: f64,
    stretch: f64,
    beam: usize,
    min_transition: f64,
}

pub fn library(server: &Server) -> Result<HashMap<i64, Summary>> {
    let list: Vec<Summary> = get(server, "/api/mix/library", Duration::from_secs(60))?;
    Ok(list.into_iter().map(|s| (s.id, s)).collect())
}

pub fn state(server: &Server, id: i64) -> Result<MixState> {
    get(server, &format!("/api/mix/state/{id}"), Duration::from_secs(20))
}

pub fn next(server: &Server, id: i64, limit: usize) -> Result<Vec<Way>> {
    get(server, &format!("/api/mix/next/{id}?limit={limit}"), Duration::from_secs(60))
}

/// Has the server analyse the track now, ahead of its queue, and waits: a
/// download and seconds of analysis, minutes when its workers are busy. The
/// server gives up after ten.
pub fn analyse_now(server: &Server, id: i64) -> Result<()> {
    let _: serde_json::Value = post(server, &format!("/api/mix/tracks/{id}/analyse"), &serde_json::Value::Null, Duration::from_secs(11 * 60))?;
    Ok(())
}

/// A set from `first`, as long as `minutes`, with the planner's own defaults.
/// Slow on a small server, half a minute for twenty minutes of set measured,
/// and longer the longer the set.
pub fn plan(server: &Server, first: i64, minutes: f64) -> Result<Plan> {
    let request = SetRequest { first: Some(first), like: None, minutes, stretch: 6.0, beam: 8, min_transition: 0.35 };
    post(server, "/api/mix/set", &request, Duration::from_secs(10 * 60))
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

fn get<T: serde::de::DeserializeOwned>(server: &Server, path: &str, timeout: Duration) -> Result<T> {
    let url = format!("{}{path}", server.base.trim_end_matches('/'));
    let response = agent(timeout)
        .get(&url)
        .header("Authorization", &format!("Bearer {}", server.token))
        .call()
        .with_context(|| format!("asking {url}"))?;
    answer(path, response)
}

fn post<T: serde::de::DeserializeOwned>(server: &Server, path: &str, body: &impl Serialize, timeout: Duration) -> Result<T> {
    let url = format!("{}{path}", server.base.trim_end_matches('/'));
    let response = agent(timeout)
        .post(&url)
        .header("Authorization", &format!("Bearer {}", server.token))
        .header("Content-Type", "application/json")
        .send(serde_json::to_string(body)?)
        .with_context(|| format!("asking {url}"))?;
    answer(path, response)
}

fn answer<T: serde::de::DeserializeOwned>(path: &str, mut response: ureq::http::Response<ureq::Body>) -> Result<T> {
    let status = response.status().as_u16();
    let body = response.body_mut().with_config().limit(64 << 20).read_to_string().unwrap_or_default();
    // The server says why in a JSON `message`, when it says.
    let why = || {
        serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["message"].as_str().map(str::to_string))
            .unwrap_or_else(|| body.chars().take(200).collect())
    };
    match status {
        200..=299 => serde_json::from_str(&body).with_context(|| format!("reading {path}")),
        401 | 403 => bail!("the server refused the token"),
        404 if path.starts_with("/api/mix/library") => bail!("this server has no mix library"),
        _ => bail!("{}", why()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against a real server, `STELLY_SERVER` and `STELLY_TOKEN` or the app's
    /// pairing: only what reads, and a plan, which stores nothing.
    #[test]
    #[ignore = "needs a Stelly server with a mix library"]
    fn reads_a_server_s_mix_library() {
        let server = Server::find().expect("a server");
        let library = library(&server).unwrap();
        assert!(!library.is_empty());
        let steady = library.values().filter(|s| s.steady).count();
        let first = library.values().filter(|s| s.steady && s.phrases > 8).map(|s| s.id).min().expect("a steady track");
        let ways = next(&server, first, 12).unwrap();
        assert!(ways.iter().all(|w| w.to != first && w.score.is_finite()));
        let unanalysed = (1..100_000).find(|id| !library.contains_key(id)).unwrap();
        let state = state(&server, unanalysed).unwrap();
        let plan = plan(&server, first, 20.0).unwrap();
        assert_eq!(plan.tracks.first(), Some(&first));
        assert_eq!(plan.ways.len(), plan.tracks.len().saturating_sub(1));
        println!(
            "{} analysed, {steady} steady; {} ways from {first}, best {:.2}; {unanalysed} is {state:?}; a set of {} tracks, {:.0} min, mean {:.2}",
            library.len(),
            ways.len(),
            ways.first().map_or(0.0, |w| w.score),
            plan.tracks.len(),
            plan.seconds / 60.0,
            plan.mean_score
        );
    }

    #[test]
    fn reads_the_server_s_shapes() {
        let summaries: Vec<Summary> = serde_json::from_str(
            r#"[{"id": 7, "artist": "A", "title": "T", "bpm": 124.02, "steady": true, "key": "8A", "seconds": 341.5, "phrases": 21}]"#,
        )
        .unwrap();
        assert_eq!(summaries[0].key.as_deref(), Some("8A"));
        let way: Way = serde_json::from_str(
            r#"{"to": 9, "score": 0.81, "sound": 0.96, "ratio": 1.02, "key_steps": 1, "energy": -0.5, "out": 12, "out_at": [200.0, 216.0], "into": 1, "into_at": [8.0, 24.0]}"#,
        )
        .unwrap();
        assert_eq!((way.to, way.out_at, way.into_at), (9, (200.0, 216.0), (8.0, 24.0)));
        let state: MixState = serde_json::from_str(r#"{"state": "queued", "reason": "liked", "ahead": 3}"#).unwrap();
        assert_eq!(state, MixState::Queued { reason: "liked".into(), ahead: 3 });
        // Every field of the set request goes, as the server wants them all.
        let sent = serde_json::to_value(SetRequest { first: Some(7), like: None, minutes: 30.0, stretch: 6.0, beam: 8, min_transition: 0.35 }).unwrap();
        for field in ["first", "like", "minutes", "stretch", "beam", "min_transition"] {
            assert!(sent.get(field).is_some(), "{field}");
        }
    }
}
