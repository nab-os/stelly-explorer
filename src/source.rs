//! Where the space comes from: a Stelly server, asked for `/api/space/vectors`
//! with a device's token, and the copy of it kept on disk.
//!
//! The server and token are the Stelly app's own when it is paired on this
//! machine, so nothing needs pairing twice; `STELLY_SERVER` and
//! `STELLY_TOKEN` override them, as they do for the app.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use stelly_core::vectors::Vectors;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Server {
    pub base: String,
    pub token: String,
}

impl Server {
    /// The environment's, else what the explorer was last given, else the
    /// Stelly app's pairing.
    pub fn find() -> Option<Server> {
        if let (Ok(base), Ok(token)) = (std::env::var("STELLY_SERVER"), std::env::var("STELLY_TOKEN")) {
            return Some(Server { base, token });
        }
        [config_dir().join("server.json"), stelly_app_dir().join("server.json")]
            .iter()
            .find_map(|path| serde_json::from_slice::<Server>(&std::fs::read(path).ok()?).ok())
            .filter(|s| !s.base.is_empty() && !s.token.is_empty())
    }

    pub fn save(&self) -> Result<()> {
        let dir = config_dir();
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("server.json"), serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

/// Where a load came from, for the status line.
#[derive(Clone, Debug, PartialEq)]
pub enum Origin {
    /// Downloaded just now, this many bytes on the wire.
    Server { bytes: usize },
    /// The server said the copy on disk is current.
    Unchanged,
    /// The server could not be asked; the copy on disk, as of its last load.
    Offline { reason: String },
}

/// The space, fresh if the server can be reached, else the last copy.
pub fn load(server: &Server) -> Result<(Vectors, Origin)> {
    let cache = cache_dir();
    let body_path = cache.join("vectors.bin");
    let tag_path = cache.join("vectors.etag");
    let held = std::fs::read(&body_path).ok();
    let tag = held.as_ref().and_then(|_| std::fs::read_to_string(&tag_path).ok());

    match fetch(server, tag.as_deref()) {
        Ok(Fetched::Body { bytes, tag, wire }) => {
            let vectors = Vectors::decode(&bytes).context("reading what the server sent")?;
            // A failed write costs the next start a download, nothing more.
            let _ = std::fs::create_dir_all(&cache)
                .and_then(|_| std::fs::write(&body_path, &bytes))
                .and_then(|_| std::fs::write(&tag_path, tag.unwrap_or_default()));
            Ok((vectors, Origin::Server { bytes: wire }))
        }
        Ok(Fetched::Unchanged) => {
            let bytes = held.context("the server says the copy is current, but there is none")?;
            Ok((Vectors::decode(&bytes)?, Origin::Unchanged))
        }
        Err(err) => match held {
            Some(bytes) => Ok((Vectors::decode(&bytes)?, Origin::Offline { reason: format!("{err:#}") })),
            None => Err(err),
        },
    }
}

enum Fetched {
    Body { bytes: Vec<u8>, tag: Option<String>, wire: usize },
    Unchanged,
}

fn fetch(server: &Server, tag: Option<&str>) -> Result<Fetched> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(180)))
        .http_status_as_error(false)
        .build()
        .into();
    let url = format!("{}/api/space/vectors", server.base.trim_end_matches('/'));
    let mut request = agent
        .get(&url)
        .header("Authorization", &format!("Bearer {}", server.token))
        .header("Accept-Encoding", "zstd");
    if let Some(tag) = tag {
        request = request.header("If-None-Match", tag);
    }
    let mut response = request.call().with_context(|| format!("asking {url}"))?;
    match response.status().as_u16() {
        304 => return Ok(Fetched::Unchanged),
        200 => {}
        401 | 403 => bail!("the server refused the token: pair this device again"),
        404 => bail!("this server has no /api/space/vectors yet: it needs Stelly with stelly-explorer support"),
        503 => bail!("the server has no space built yet"),
        status => bail!("the server answered {status}"),
    }
    let tag = response.headers().get("etag").and_then(|v| v.to_str().ok()).map(str::to_string);
    let zstd = response
        .headers()
        .get("content-encoding")
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"zstd"));
    let wire = response
        .body_mut()
        .with_config()
        .limit(1 << 30)
        .read_to_vec()
        .context("reading the space")?;
    let size = wire.len();
    let bytes = if zstd { zstd::decode_all(&wire[..]).context("unpacking the space")? } else { wire };
    Ok(Fetched::Body { bytes, tag, wire: size })
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

fn xdg(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| home().join(fallback))
}

fn config_dir() -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config").join("stelly-explorer")
}

fn cache_dir() -> PathBuf {
    xdg("XDG_CACHE_HOME", ".cache").join("stelly-explorer")
}

/// The Stelly app's data folder, where it keeps its pairing.
fn stelly_app_dir() -> PathBuf {
    std::env::var_os("STELLY_CLIENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| xdg("XDG_DATA_HOME", ".local/share").join("stelly"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against a real server, given by `STELLY_SERVER` and `STELLY_TOKEN`:
    /// a download, then the same again answered from the cache.
    #[test]
    #[ignore = "needs a Stelly server"]
    fn loads_from_a_server_then_from_the_cache() {
        let server = Server::find().expect("STELLY_SERVER and STELLY_TOKEN");
        let (first, origin) = load(&server).unwrap();
        assert!(matches!(origin, Origin::Server { .. }), "{origin:?}");
        assert!(first.head.n > 0 && first.head.dims > 0);
        let (second, origin) = load(&server).unwrap();
        assert_eq!(origin, Origin::Unchanged);
        assert_eq!(first.ids, second.ids);

        let wrong = Server { base: server.base.clone(), token: "nope".into() };
        let (_, origin) = load(&wrong).unwrap();
        assert!(matches!(origin, Origin::Offline { ref reason } if reason.contains("refused")), "{origin:?}");
    }
}
