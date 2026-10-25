//! The window: the space drawn in the middle, how to project and colour it
//! on the left, the picked track on the right, a status line at the bottom.
//!
//! Anything slow, loading the space or projecting it, runs on a thread of
//! its own and hands its result back through a channel; the frame only
//! checks for it, so the view keeps turning while UMAP runs.

use crate::camera::Camera;
use crate::colour::{self, Colouring, Legend, Rgba};
use crate::labels::{self, Region};
use crate::mix::{self, Plan, Readiness, Summary, Way};
use crate::phrase_paths::{self, Placement};
use crate::project::{self, Progress, Projection};
use crate::render::{Draw, Instance, Lines, Points, Renderer};
use crate::source::{self, Origin, Server};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use eframe::egui_wgpu;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::Arc;
use std::time::Instant;
use stelly_core::api::MixState;
use stelly_core::phrases::Phrases;
use stelly_core::vectors::{Vectors, unquantise};

const BACKGROUND: Color32 = Color32::from_rgb(0x12, 0x13, 0x1a);
const PANEL: Color32 = Color32::from_rgb(0x1b, 0x1d, 0x26);
const TEXT: Color32 = Color32::from_rgb(0xe5, 0xe7, 0xef);
const MUTED: Color32 = Color32::from_rgb(0x8b, 0x90, 0xa3);
const ACCENT: Color32 = Color32::from_rgb(0x7a, 0xa2, 0xf7);

/// How far from a point the pointer may be and still be over it, in points.
const REACH: f32 = 9.0;
/// The most names on screen at once.
const LABELS: usize = 40;
const NEIGHBOURS: usize = 15;
/// Ways out of the picked track asked of the server.
const WAYS: usize = 12;

struct Data {
    v: Arc<Vectors>,
    weights: Vec<f32>,
    unit: Arc<Vec<f32>>,
    /// Title, artist and album, lowercased once, for the search.
    haystacks: Vec<String>,
    /// Track id to row.
    row_of: HashMap<i64, usize>,
}

impl Data {
    fn new(v: Vectors) -> Self {
        let weights: Vec<f32> = v.head.blocks.iter().map(|b| b.default_weight).collect();
        let unit = Arc::new(project::weighted(&v, &weights));
        let haystacks = (0..v.head.n).map(|i| format!("{}\n{}\n{}", v.titles[i], v.artists[i], v.albums[i]).to_lowercase()).collect();
        let row_of = v.ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        Self { v: Arc::new(v), weights, unit, haystacks, row_of }
    }
}

struct Job {
    projection: Projection,
    progress: Progress,
    result: Receiver<Option<Vec<[f32; 3]>>>,
    started: Instant,
}

/// What a call to the server's mix library came back with. Each is for the
/// track it names, and dropped if that is no longer the one picked.
enum MixReply {
    Library(anyhow::Result<HashMap<i64, Summary>>),
    State(i64, anyhow::Result<MixState>),
    Ways(i64, anyhow::Result<Vec<Way>>),
    Analysed(i64, anyhow::Result<()>),
    Planned(i64, anyhow::Result<Plan>),
}

/// What a click on a legend narrows the view to.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Focus {
    Genre(u16),
    /// A Camelot place, and the keys that mix with it.
    Key(u8, bool),
    Mix(Readiness),
}

/// The projection panel's own choices, applied on change (or, for UMAP,
/// which takes a while, on asking).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Map,
    Pca,
    Axes,
    Umap,
}

pub struct Explorer {
    server: Option<Server>,
    form: Server,
    loading: Option<Receiver<anyhow::Result<(Vectors, Origin)>>>,
    failure: Option<String>,
    origin: Option<Origin>,
    data: Option<Data>,

    kind: Kind,
    three: bool,
    axes: [Option<usize>; 3],
    spread: bool,
    sample: usize,
    neighbours_k: usize,
    projection: Projection,
    job: Option<Job>,
    took: Option<f32>,
    positions: Vec<[f32; 3]>,
    regions: Vec<Region>,

    colouring: Colouring,
    colours: Vec<Rgba>,
    legend: Legend,

    search: String,
    matches: Option<Vec<bool>>,
    focus: Option<Focus>,

    radius: f32,
    opacity: f32,
    show_labels: bool,
    show_links: bool,

    camera: Camera,
    points: Arc<Points>,
    /// Instance i is track `drawn[i]`.
    drawn: Vec<u32>,
    dirty: bool,
    /// Lit tracks' places on screen, for picking: (track, x, y, depth).
    screen: Vec<(u32, f32, f32, f32)>,
    screen_for: Option<([f32; 16], Rect)>,
    hover: Option<usize>,
    selected: Option<usize>,
    neighbours: Vec<(usize, f32)>,
    frame_times: Vec<f32>,
    last_frame: Instant,

    mix_tx: Sender<MixReply>,
    mix_rx: Receiver<MixReply>,
    /// stelly-mix's library on the server, by track id, once read.
    mix: Option<HashMap<i64, Summary>>,
    mix_failure: Option<String>,
    /// Each row's readiness; empty until the mix library comes.
    readiness: Vec<Readiness>,
    /// The track the picked one's mix data was asked for.
    asked_for: Option<i64>,
    mix_state: Option<MixState>,
    ways: Option<anyhow::Result<Vec<Way>>>,
    analysing: Option<i64>,
    analyse_failure: Option<String>,
    set_minutes: f64,
    /// When the plan being waited for was asked.
    planning: Option<Instant>,
    set: Option<anyhow::Result<Plan>>,
    show_ways: bool,

    /// The global switch: every mix-analysed track drawn as its phrases' path.
    show_paths: bool,
    path_width: f32,
    phrases: Option<Arc<Phrases>>,
    phrases_loading: Option<Receiver<anyhow::Result<(Phrases, Origin)>>>,
    paths_failure: Option<String>,
    placing: Option<(Progress, Receiver<Option<Placement>>)>,
    placement: Option<Arc<Placement>>,
    /// Row to its index in `placement.tracks`.
    path_of: HashMap<usize, usize>,
    phrase_positions: Vec<[f32; 3]>,
    lines: Arc<Lines>,
}

impl Explorer {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        if let Some(state) = cc.wgpu_render_state.as_ref() {
            Renderer::install(state);
        }
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = PANEL;
        visuals.window_fill = PANEL;
        visuals.extreme_bg_color = BACKGROUND;
        visuals.selection.bg_fill = ACCENT.linear_multiply(0.45);
        cc.egui_ctx.set_visuals(visuals);

        let server = Server::find();
        let (mix_tx, mix_rx) = channel();
        let mut app = Self {
            form: server.clone().unwrap_or_default(),
            server,
            loading: None,
            failure: None,
            origin: None,
            data: None,
            kind: Kind::Map,
            three: false,
            axes: [None, None, None],
            spread: true,
            sample: 25_000,
            neighbours_k: 15,
            projection: Projection::Map,
            job: None,
            took: None,
            positions: Vec::new(),
            regions: Vec::new(),
            colouring: Colouring::Genre,
            colours: Vec::new(),
            legend: Legend::Categories(Vec::new()),
            search: String::new(),
            matches: None,
            focus: None,
            radius: 1.6,
            opacity: 0.9,
            show_labels: true,
            show_links: true,
            camera: Camera::default(),
            points: Arc::new(Points { version: 0, instances: Vec::new() }),
            drawn: Vec::new(),
            dirty: false,
            screen: Vec::new(),
            screen_for: None,
            hover: None,
            selected: None,
            neighbours: Vec::new(),
            frame_times: Vec::new(),
            last_frame: Instant::now(),
            mix_tx,
            mix_rx,
            mix: None,
            mix_failure: None,
            readiness: Vec::new(),
            asked_for: None,
            mix_state: None,
            ways: None,
            analysing: None,
            analyse_failure: None,
            set_minutes: 30.0,
            planning: None,
            set: None,
            show_ways: true,
            show_paths: false,
            path_width: 1.2,
            phrases: None,
            phrases_loading: None,
            paths_failure: None,
            placing: None,
            placement: None,
            path_of: HashMap::new(),
            phrase_positions: Vec::new(),
            lines: Arc::new(Lines { version: 0, lines: Vec::new() }),
        };
        app.load(&cc.egui_ctx);
        app
    }

    fn load(&mut self, ctx: &egui::Context) {
        let Some(server) = self.server.clone() else { return };
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(source::load(&server));
            ctx.request_repaint();
        });
        self.loading = Some(rx);
        self.failure = None;
    }

    fn arrived(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.loading else { return };
        let Ok(result) = rx.try_recv() else { return };
        self.loading = None;
        match result {
            Ok((v, origin)) => {
                let keep = self.selected.and_then(|i| self.data.as_ref().map(|d| d.v.ids[i]));
                let data = Data::new(v);
                // Default axes: the first two mood columns, where there are any.
                let mood = data.v.head.blocks.iter().find(|b| b.name == "mood").map(|b| b.start);
                self.axes = [mood.or(Some(0)), mood.map(|m| m + 1).or(Some(1)), None];
                self.selected = keep.and_then(|id| data.v.ids.iter().position(|x| *x == id));
                self.data = Some(data);
                self.origin = Some(origin);
                self.recolour();
                self.reproject(ctx, self.projection.clone());
                self.find_neighbours();
                self.update_readiness();
                self.ask(ctx, |s| MixReply::Library(mix::library(s)));
            }
            Err(err) => self.failure = Some(format!("{err:#}")),
        }
    }

    /// One call to the server's mix library, on a thread of its own.
    fn ask(&self, ctx: &egui::Context, call: impl FnOnce(&Server) -> MixReply + Send + 'static) {
        let Some(server) = self.server.clone() else { return };
        let (tx, ctx) = (self.mix_tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(call(&server));
            ctx.request_repaint();
        });
    }

    /// Whatever the mix library has answered since the last frame.
    fn mix_replies(&mut self, ctx: &egui::Context) {
        while let Ok(reply) = self.mix_rx.try_recv() {
            let picked = self.picked_id();
            match reply {
                MixReply::Library(Ok(library)) => {
                    self.mix = Some(library);
                    self.mix_failure = None;
                    self.update_readiness();
                    // The picked track may have just become analysed.
                    self.asked_for = None;
                }
                MixReply::Library(Err(err)) => self.mix_failure = Some(format!("{err:#}")),
                MixReply::State(id, state) if Some(id) == picked => self.mix_state = state.ok(),
                MixReply::Ways(id, ways) if Some(id) == picked => self.ways = Some(ways),
                MixReply::Analysed(id, result) => {
                    self.analysing = None;
                    match result {
                        Ok(()) => {
                            self.analyse_failure = None;
                            self.ask(ctx, |s| MixReply::Library(mix::library(s)));
                        }
                        Err(err) if Some(id) == picked => self.analyse_failure = Some(format!("{err:#}")),
                        Err(_) => {}
                    }
                }
                MixReply::Planned(id, plan) if Some(id) == picked => {
                    self.planning = None;
                    self.set = Some(plan);
                }
                MixReply::Planned(..) => self.planning = None,
                _ => {}
            }
        }
        // The picked track's state and ways, asked for once per pick.
        if picked_changed(self.picked_id(), &mut self.asked_for) {
            self.mix_state = None;
            self.ways = None;
            self.analyse_failure = None;
            if let Some(id) = self.asked_for {
                if self.summary_of(id).is_some() {
                    self.ask(ctx, move |s| MixReply::Ways(id, mix::next(s, id, WAYS)));
                } else if self.mix.is_some() {
                    self.ask(ctx, move |s| MixReply::State(id, mix::state(s, id)));
                }
            }
        }
    }

    /// The phrase paths, turned on: what is not there yet is asked for.
    fn paths_on(&mut self, ctx: &egui::Context) {
        if self.phrases.is_none() && self.phrases_loading.is_none() {
            let Some(server) = self.server.clone() else { return };
            let (tx, rx) = channel();
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let _ = tx.send(source::phrases(&server));
                ctx.request_repaint();
            });
            self.phrases_loading = Some(rx);
            self.paths_failure = None;
        } else if self.placement.is_none() && self.placing.is_none() {
            self.place(ctx);
        }
        self.dirty = true;
    }

    /// Where each phrase sits among the tracks, on a thread: seconds over a
    /// whole library. Again whenever the weights change.
    fn place(&mut self, ctx: &egui::Context) {
        let (Some(data), Some(phrases)) = (&self.data, &self.phrases) else { return };
        if let Some((progress, _)) = self.placing.take() {
            progress.cancelled.store(true, Ordering::Relaxed);
        }
        let progress = Progress::default();
        let (tx, rx) = channel();
        let (v, unit, weights, phrases, row_of, pr, ctx) =
            (data.v.clone(), data.unit.clone(), data.weights.clone(), phrases.clone(), data.row_of.clone(), progress.clone(), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(phrase_paths::place(&v, &unit, &weights, &phrases, &row_of, &pr));
            ctx.request_repaint();
        });
        self.placing = Some((progress, rx));
    }

    fn paths_arrived(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.phrases_loading
            && let Ok(result) = rx.try_recv()
        {
            self.phrases_loading = None;
            match result {
                Ok((phrases, _)) => {
                    self.phrases = Some(Arc::new(phrases));
                    if self.show_paths {
                        self.place(ctx);
                    }
                }
                Err(err) => self.paths_failure = Some(format!("{err:#}")),
            }
        }
        if let Some((_, rx)) = &self.placing
            && let Ok(result) = rx.try_recv()
        {
            self.placing = None;
            if let Some(placement) = result {
                self.path_of = placement.tracks.iter().enumerate().map(|(k, (row, _))| (*row, k)).collect();
                self.placement = Some(Arc::new(placement));
                self.place_phrases();
            }
        }
    }

    /// The phrases' places in the projection shown: where their nearest
    /// tracks are, so nothing to work out again but a weighted mean.
    fn place_phrases(&mut self) {
        self.phrase_positions = match &self.placement {
            Some(placement) if !self.positions.is_empty() => placement.positions(&self.positions),
            _ => Vec::new(),
        };
        self.dirty = true;
    }

    fn picked_id(&self) -> Option<i64> {
        Some(self.data.as_ref()?.v.ids[self.selected?])
    }

    fn summary_of(&self, id: i64) -> Option<&Summary> {
        self.mix.as_ref()?.get(&id)
    }

    fn update_readiness(&mut self) {
        let (Some(data), Some(library)) = (&self.data, &self.mix) else { return };
        self.readiness = data
            .v
            .ids
            .iter()
            .map(|id| match library.get(id) {
                Some(s) if s.steady => Readiness::Ready,
                Some(_) => Readiness::Drifting,
                None => Readiness::NotAnalysed,
            })
            .collect();
        if self.colouring == Colouring::Mix {
            self.recolour();
        }
        self.dirty = true;
    }

    fn reproject(&mut self, ctx: &egui::Context, projection: Projection) {
        let Some(data) = &self.data else { return };
        if let Some(job) = self.job.take() {
            job.progress.cancelled.store(true, Ordering::Relaxed);
        }
        let progress = Progress::default();
        let (tx, rx) = channel();
        let (v, unit, p, pr, ctx) = (data.v.clone(), data.unit.clone(), projection.clone(), progress.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = project::run(&p, &v, &unit, &pr);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        self.job = Some(Job { projection, progress, result: rx, started: Instant::now() });
    }

    fn projected(&mut self) {
        let Some(job) = &self.job else { return };
        let Ok(result) = job.result.try_recv() else { return };
        let job = self.job.take().unwrap();
        let Some(positions) = result else { return };
        let Some(data) = &self.data else { return };
        let dims_changed = job.projection.dims() != self.projection.dims() || self.positions.is_empty();
        let v = &data.v;
        self.regions = labels::find(&positions, &v.genre, v.head.genres.len(), job.projection.dims(), |g| {
            v.head.genres.get(g as usize).is_some_and(|name| !name.is_empty())
        });
        self.positions = positions;
        self.took = Some(job.started.elapsed().as_secs_f32());
        if dims_changed || job.projection != self.projection {
            self.camera = Camera::default();
        }
        self.projection = job.projection;
        self.place_phrases();
        self.dirty = true;
    }

    fn recolour(&mut self) {
        let Some(data) = &self.data else { return };
        (self.colours, self.legend) = colour::colour(&data.v, self.colouring, &self.readiness);
        self.dirty = true;
    }

    fn lit(&self, i: usize) -> bool {
        let Some(data) = &self.data else { return false };
        if self.matches.as_ref().is_some_and(|m| !m[i]) {
            return false;
        }
        match self.focus {
            None => true,
            Some(Focus::Genre(g)) => data.v.genre[i] == g,
            Some(Focus::Key(number, minor)) => {
                let Some(key) = data.v.key(i).filter(|_| data.v.key_strength[i] >= colour::KEY_SURE) else { return false };
                let (n, letter) = key.camelot();
                let a = letter == 'A';
                (n == number && a != minor) || (a == minor && (n == number || (n % 12) + 1 == number || (number % 12) + 1 == n))
            }
            Some(Focus::Mix(r)) => self.readiness.get(i).copied().unwrap_or(Readiness::NotAnalysed) == r,
        }
    }

    /// The instance buffer, from positions, colours and what is lit. The dim
    /// first, so in 2D the lit draw over them.
    fn rebuild(&mut self) {
        if !self.dirty || self.data.is_none() {
            return;
        }
        self.dirty = false;
        let n = self.positions.len().min(self.colours.len());
        let filtering = self.matches.is_some() || self.focus.is_some();
        let mut dim = Vec::new();
        let mut lit = Vec::new();
        for i in 0..n {
            let p = self.positions[i];
            if !p[0].is_finite() {
                continue;
            }
            if self.lit(i) { lit.push(i as u32) } else { dim.push(i as u32) }
        }
        let lit_alpha = (self.opacity * 255.0) as u8;
        let dim_alpha = if filtering { 22 } else { lit_alpha };
        let mut instances = Vec::with_capacity(dim.len() + lit.len());
        for (list, alpha) in [(&dim, dim_alpha), (&lit, lit_alpha)] {
            for &i in list.iter() {
                let mut colour = self.colours[i as usize];
                colour[3] = alpha;
                instances.push(Instance { position: self.positions[i as usize], colour });
            }
        }
        self.drawn = dim.into_iter().chain(lit).collect();
        self.points = Arc::new(Points { version: self.points.version + 1, instances });
        self.screen_for = None;

        // The paths of the lit tracks, faint, in their tracks' colours.
        let lines = match &self.placement {
            Some(placement) if self.show_paths && !self.phrase_positions.is_empty() => {
                let alpha = if filtering { 150 } else { 70 };
                placement.lines(&self.phrase_positions, |row| {
                    self.lit(row).then(|| {
                        let mut c = self.colours.get(row).copied().unwrap_or(colour::MUTED);
                        c[3] = alpha;
                        c
                    })
                })
            }
            _ => Vec::new(),
        };
        self.lines = Arc::new(Lines { version: self.lines.version + 1, lines });
    }

    fn find_neighbours(&mut self) {
        self.neighbours.clear();
        let (Some(data), Some(i)) = (&self.data, self.selected) else { return };
        let d = data.v.head.dims;
        let row = &data.unit[i * d..(i + 1) * d];
        let mut scored: Vec<(usize, f32)> = data
            .unit
            .chunks_exact(d)
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, other)| (j, row.iter().zip(other).map(|(a, b)| a * b).sum()))
            .collect();
        let k = NEIGHBOURS.min(scored.len());
        if k == 0 {
            return;
        }
        scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.neighbours = scored;
    }

    fn select(&mut self, i: Option<usize>) {
        if self.selected != i {
            self.selected = i;
            self.find_neighbours();
            // A set belongs to the track it was planned from.
            self.set = None;
            self.planning = None;
        }
    }

    fn three(&self) -> bool {
        self.projection.dims() == 3
    }

    fn desired(&self) -> Projection {
        let dims = if self.three { 3 } else { 2 };
        match self.kind {
            Kind::Map => Projection::Map,
            Kind::Pca => Projection::Pca { dims },
            Kind::Axes => Projection::Axes { columns: [self.axes[0], self.axes[1], if self.three { self.axes[2].or(self.axes[1].map(|c| c + 1)) } else { None }], spread: self.spread },
            Kind::Umap => Projection::Umap { dims, sample: self.sample, neighbours: self.neighbours_k },
        }
    }
}

impl eframe::App for Explorer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.arrived(&ctx);
        self.mix_replies(&ctx);
        self.paths_arrived(&ctx);
        self.projected();
        let now = Instant::now();
        self.frame_times.push((now - self.last_frame).as_secs_f32());
        self.last_frame = now;
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        if self.job.is_some() || self.loading.is_some() || self.analysing.is_some() || self.planning.is_some() || self.phrases_loading.is_some() || self.placing.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::Panel::bottom("status").show(ui, |ui| self.status(ui));
        egui::Panel::left("controls").default_size(300.0).resizable(true).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, &ctx));
        });
        if self.selected.is_some() && self.data.is_some() {
            egui::Panel::right("track").default_size(320.0).resizable(true).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.details(ui));
            });
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(BACKGROUND)).show(ui, |ui| {
            if self.data.is_some() {
                self.view(ui);
            } else {
                self.welcome(ui, &ctx);
            }
        });
    }
}

// ------------------------------------------------------------------ panels

impl Explorer {
    fn welcome(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(60.0);
        ui.vertical_centered(|ui| {
            ui.set_max_width(420.0);
            ui.heading(RichText::new("Stelly Explorer").color(TEXT).size(24.0));
            ui.add_space(8.0);
            if self.loading.is_some() {
                ui.label(RichText::new("Reading the space from the server…").color(MUTED));
                ui.spinner();
                return;
            }
            if let Some(failure) = &self.failure {
                ui.label(RichText::new(failure).color(Color32::from_rgb(0xf7, 0x76, 0x8e)));
                ui.add_space(8.0);
            }
            ui.label(RichText::new("Pair it with a Stelly server: its address, and a device token from\nstelly-server pair --name explorer --scope play").color(MUTED));
            ui.add_space(12.0);
            egui::Grid::new("setup").num_columns(2).spacing([10.0, 8.0]).show(ui, |ui| {
                ui.label("Server");
                ui.add(egui::TextEdit::singleline(&mut self.form.base).hint_text("https://music.example.org").desired_width(280.0));
                ui.end_row();
                ui.label("Token");
                ui.add(egui::TextEdit::singleline(&mut self.form.token).password(true).desired_width(280.0));
                ui.end_row();
            });
            ui.add_space(8.0);
            let ready = !self.form.base.trim().is_empty() && !self.form.token.trim().is_empty();
            if ui.add_enabled(ready, egui::Button::new("Connect")).clicked() {
                let server = Server { base: self.form.base.trim().to_string(), token: self.form.token.trim().to_string() };
                if let Err(err) = server.save() {
                    self.failure = Some(format!("could not save the pairing: {err:#}"));
                }
                self.server = Some(server);
                self.load(ctx);
            }
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(6.0);
        ui.heading(RichText::new("Stelly Explorer").color(TEXT));
        let Some(v) = self.data.as_ref().map(|d| d.v.clone()) else {
            ui.label(RichText::new("No space loaded yet.").color(MUTED));
            return;
        };
        let n = v.head.n;
        let on_map = v.xy.iter().filter(|p| p[0].is_finite()).count();
        ui.label(RichText::new(format!("{n} tracks, {on_map} on the map, {} genres", v.head.genres.len())).color(MUTED));
        match (&self.mix, &self.mix_failure) {
            (Some(library), _) => {
                let ready = self.readiness.iter().filter(|r| **r == Readiness::Ready).count();
                ui.label(RichText::new(format!("{} analysed for mixing, {ready} ready", library.len())).color(MUTED));
            }
            (None, Some(failure)) => {
                ui.label(RichText::new(format!("No mix data: {failure}")).color(MUTED).small());
            }
            (None, None) => {
                ui.label(RichText::new("Reading the mix library…").color(MUTED).small());
            }
        }
        ui.horizontal(|ui| {
            if let Some(server) = &self.server {
                ui.label(RichText::new(server.base.trim_start_matches("https://").trim_start_matches("http://")).color(MUTED).small());
            }
            if ui.add_enabled(self.loading.is_none(), egui::Button::new("Reload").small()).clicked() {
                self.load(ctx);
            }
        });
        if let Some(Origin::Offline { reason }) = &self.origin {
            ui.label(RichText::new(format!("Offline, showing the last copy: {reason}")).color(Color32::from_rgb(0xe0, 0xaf, 0x68)).small());
        }

        // -------------------------------------------------- projection
        ui.separator();
        section(ui, "Projection");
        let before = (self.kind, self.three, self.axes, self.spread);
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.kind, Kind::Map, "Map").on_hover_text("The server's own map: UMAP of the whole catalogue");
            ui.selectable_value(&mut self.kind, Kind::Pca, "PCA").on_hover_text("The directions the space varies most along");
            ui.selectable_value(&mut self.kind, Kind::Axes, "Axes").on_hover_text("Any columns of the space as the axes");
            ui.selectable_value(&mut self.kind, Kind::Umap, "UMAP").on_hover_text("UMAP run here, on a sample, in 2D or 3D");
        });
        if self.kind != Kind::Map {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.three, false, "2D");
                ui.selectable_value(&mut self.three, true, "3D");
            });
        }
        if self.kind == Kind::Axes {
            let names = ["Across", "Up", "Depth"];
            let count = if self.three { 3 } else { 2 };
            for (axis, name) in names.iter().enumerate().take(count) {
                column_picker(ui, &v, name, &mut self.axes[axis], axis == 2);
            }
            ui.checkbox(&mut self.spread, "Spread evenly").on_hover_text("Place by rank, so the tracks fill each axis instead of piling up in the middle");
        }
        let mut run_umap = false;
        if self.kind == Kind::Umap {
            ui.add(egui::Slider::new(&mut self.sample, 1000.min(n)..=n.max(1000)).logarithmic(true).text("tracks"));
            ui.add(egui::Slider::new(&mut self.neighbours_k, 5..=50).text("neighbours"));
            // Measured at 25,000 tracks on 32 threads: about 4 s of neighbour
            // search, quadratic, and 2.5 s of layout, linear.
            let share = self.sample as f32 / 25_000.0;
            let seconds = share * share * 4.0 + share * 2.5;
            ui.label(RichText::new(format!("Roughly {seconds:.0} s here; the rest of the tracks are left out.")).color(MUTED).small());
            run_umap = ui.button("Run UMAP").clicked();
        }
        let changed = before != (self.kind, self.three, self.axes, self.spread);
        if (changed && self.kind != Kind::Umap) || run_umap {
            let p = self.desired();
            self.reproject(ctx, p);
        }
        if let Some(job) = &self.job {
            let done = job.progress.done.load(Ordering::Relaxed) as f32 / 1000.0;
            ui.horizontal(|ui| {
                ui.add(egui::ProgressBar::new(done).desired_width(180.0).show_percentage());
                if ui.small_button("Stop").clicked() {
                    job.progress.cancelled.store(true, Ordering::Relaxed);
                }
            });
        }

        // ------------------------------------------------------ colour
        ui.separator();
        section(ui, "Colour");
        let before = self.colouring;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.colouring, Colouring::Genre, "Genre");
            ui.selectable_value(&mut self.colouring, Colouring::Key, "Key");
            ui.selectable_value(&mut self.colouring, Colouring::Tempo, "Tempo");
            ui.selectable_value(&mut self.colouring, Colouring::Mix, "Mix").on_hover_text("Which tracks stelly-mix has analysed whole, and which a set can use");
            let column = if let Colouring::Column(c) = self.colouring { Some(c) } else { None };
            if ui.selectable_label(column.is_some(), "Column").clicked() && column.is_none() {
                let mood = v.head.blocks.iter().find(|b| b.name == "mood").map_or(0, |b| b.start);
                self.colouring = Colouring::Column(mood);
            }
        });
        if let Colouring::Column(c) = self.colouring {
            let mut pick = Some(c);
            column_picker(ui, &v, "Column", &mut pick, false);
            if let Some(p) = pick {
                self.colouring = Colouring::Column(p);
            }
        }
        if before != self.colouring {
            self.focus = None;
            self.recolour();
        }
        self.legend(ui);

        // -------------------------------------------------------- find
        ui.separator();
        section(ui, "Find");
        let response = ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("title, artist or album").desired_width(f32::INFINITY));
        if response.changed() {
            let words: Vec<String> = self.search.to_lowercase().split_whitespace().map(str::to_string).collect();
            self.matches = match &self.data {
                Some(data) if !words.is_empty() => Some(data.haystacks.iter().map(|h| words.iter().all(|w| h.contains(w))).collect()),
                _ => None,
            };
            self.dirty = true;
        }
        if let Some(m) = &self.matches {
            let found = m.iter().filter(|x| **x).count();
            ui.label(RichText::new(format!("{found} tracks match")).color(MUTED).small());
        }
        if self.focus.is_some() && ui.button("Show everything again").clicked() {
            self.focus = None;
            self.dirty = true;
        }

        // ----------------------------------------------------- display
        ui.separator();
        section(ui, "Display");
        if ui.add(egui::Slider::new(&mut self.radius, 0.4..=6.0).text("point size")).changed() {
            self.screen_for = None;
        }
        if ui.add(egui::Slider::new(&mut self.opacity, 0.5..=1.0).text("opacity")).changed() {
            self.dirty = true;
        }
        ui.checkbox(&mut self.show_labels, "Genre names");
        ui.checkbox(&mut self.show_links, "Lines to the picked track's nearest");
        ui.checkbox(&mut self.show_ways, "Arrows to its best mixes");
        let before = self.show_paths;
        ui.checkbox(&mut self.show_paths, "Phrase paths (P)")
            .on_hover_text("Every track stelly-mix has analysed, drawn as the path its eight-bar phrases take through the space");
        if self.show_paths != before {
            if self.show_paths {
                self.paths_on(ctx);
            }
            self.dirty = true;
        }
        if self.show_paths {
            if self.phrases_loading.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("Reading the phrases…").color(MUTED).small());
                });
            }
            if let Some((progress, _)) = &self.placing {
                let done = progress.done.load(Ordering::Relaxed) as f32 / 1000.0;
                ui.add(egui::ProgressBar::new(done).desired_width(200.0).text("placing the phrases"));
            }
            if let Some(failure) = &self.paths_failure {
                ui.label(RichText::new(format!("No phrases: {failure}")).color(MUTED).small());
            }
            if let Some(phrases) = &self.phrases {
                ui.label(RichText::new(format!("{} tracks, {} phrases", phrases.head.tracks, phrases.head.phrases)).color(MUTED).small());
            }
            ui.add(egui::Slider::new(&mut self.path_width, 0.5..=4.0).text("path width"));
        }
        if ui.button("Fit the view").clicked() {
            self.camera = Camera::default();
        }

        // ----------------------------------------------------- weights
        ui.separator();
        let mut reweigh = false;
        egui::CollapsingHeader::new(RichText::new("Weights").color(TEXT).strong()).show(ui, |ui| {
            ui.label(RichText::new("How much each part of the space counts, for PCA, UMAP and the nearest tracks. The server's own defaults to start.").color(MUTED).small());
            let blocks = v.head.blocks.clone();
            for (b, w) in blocks.iter().zip(self.weights_mut()) {
                reweigh |= ui.add(egui::Slider::new(w, 0.0..=3.0).text(&b.name)).drag_stopped();
            }
            if ui.button("Server's defaults").clicked() {
                for (b, w) in blocks.iter().zip(self.weights_mut()) {
                    *w = b.default_weight;
                }
                reweigh = true;
            }
        });
        if reweigh {
            if let Some(data) = &mut self.data {
                data.unit = Arc::new(project::weighted(&data.v, &data.weights));
            }
            self.find_neighbours();
            // Where a phrase sits among the tracks depends on the weights.
            self.placement = None;
            self.phrase_positions.clear();
            if self.show_paths {
                self.place(ctx);
            }
            if matches!(self.projection, Projection::Pca { .. } | Projection::Umap { .. }) {
                let p = self.projection.clone();
                self.reproject(ctx, p);
            }
        }
        ui.add_space(12.0);
    }

    fn weights_mut(&mut self) -> &mut [f32] {
        self.data.as_mut().map_or(&mut [], |d| d.weights.as_mut_slice())
    }

    fn legend(&mut self, ui: &mut egui::Ui) {
        let mut focus = self.focus;
        match &self.legend {
            Legend::Categories(rows) => {
                let names = self.data.as_ref().map(|d| d.v.head.genres.clone()).unwrap_or_default();
                for (name, rgba, count) in rows {
                    let genre = names.iter().position(|g| g == name).map(|g| g as u16);
                    let on = genre.is_some() && focus == genre.map(Focus::Genre);
                    let response = ui
                        .horizontal(|ui| {
                            swatch(ui, *rgba);
                            let r = ui.selectable_label(on, RichText::new(name).color(TEXT));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(count.to_string()).color(MUTED).small());
                            });
                            r
                        })
                        .inner;
                    if response.clicked()
                        && let Some(g) = genre {
                            focus = if on { None } else { Some(Focus::Genre(g)) };
                        }
                }
                ui.label(RichText::new("Click a genre to show only its tracks.").color(MUTED).small());
            }
            Legend::Wheel { major, minor, unsure } => {
                egui::Grid::new("wheel").num_columns(6).spacing([4.0, 4.0]).show(ui, |ui| {
                    for (letter, row_colours, is_minor) in [('B', major, false), ('A', minor, true)] {
                        for half in 0..2 {
                            for k in 0..6 {
                                let number = (half * 6 + k + 1) as u8;
                                let on = focus == Some(Focus::Key(number, is_minor));
                                let [r, g, b, _] = row_colours[number as usize - 1];
                                let fill = Color32::from_rgb(r, g, b);
                                let text = RichText::new(format!("{number}{letter}")).color(BACKGROUND).strong().size(11.0);
                                let button = egui::Button::new(text).fill(fill).min_size(Vec2::new(36.0, 20.0)).stroke(if on { Stroke::new(2.0, TEXT) } else { Stroke::NONE });
                                if ui.add(button).on_hover_text(colour::camelot_name(number, is_minor)).clicked() {
                                    focus = if on { None } else { Some(Focus::Key(number, is_minor)) };
                                }
                            }
                            ui.end_row();
                        }
                    }
                });
                let text = match focus {
                    Some(Focus::Key(number, minor)) => {
                        let (letter, other) = if minor { ('A', 'B') } else { ('B', 'A') };
                        let prev = (number + 10) % 12 + 1;
                        let next = number % 12 + 1;
                        format!("{number}{letter} {} mixes with {prev}{letter}, {next}{letter} and {number}{other}.", colour::camelot_name(number, minor))
                    }
                    _ => format!("B is major, A minor; neighbouring numbers mix well. Grey: key unclear ({unsure} tracks)."),
                };
                ui.label(RichText::new(text).color(MUTED).small());
            }
            Legend::Mix(rows) => {
                for (name, rgba, count, readiness) in rows {
                    let on = focus == Some(Focus::Mix(*readiness));
                    let response = ui
                        .horizontal(|ui| {
                            swatch(ui, *rgba);
                            let r = ui.selectable_label(on, RichText::new(name).color(TEXT));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(count.to_string()).color(MUTED).small());
                            });
                            r
                        })
                        .inner;
                    if response.clicked() {
                        focus = if on { None } else { Some(Focus::Mix(*readiness)) };
                    }
                }
                let text = if self.mix.is_some() {
                    "Ready: analysed whole, on one tempo throughout, so a set can use it. A drifting tempo keeps a track out of planned sets."
                } else {
                    "The mix library has not come yet: every track shows as not analysed."
                };
                ui.label(RichText::new(text).color(MUTED).small());
            }
            Legend::Ramp { stops, low, middle, high } => {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 14.0), Sense::hover());
                let w = rect.width() / stops.len() as f32;
                for (k, [r, g, b, _]) in stops.iter().enumerate() {
                    let x = rect.left() + k as f32 * w;
                    ui.painter().rect_filled(Rect::from_min_max(Pos2::new(x, rect.top()), Pos2::new(x + w + 0.5, rect.bottom())), 0.0, Color32::from_rgb(*r, *g, *b));
                }
                ui.horizontal(|ui| {
                    ui.label(RichText::new(low).color(MUTED).small());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(high).color(MUTED).small());
                        ui.centered_and_justified(|ui| ui.label(RichText::new(middle).color(MUTED).small()));
                    });
                });
            }
        }
        if focus != self.focus {
            self.focus = focus;
            self.dirty = true;
        }
    }

    fn details(&mut self, ui: &mut egui::Ui) {
        let (Some(data), Some(i)) = (&self.data, self.selected) else { return };
        let v = data.v.clone();
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Track").color(MUTED).small());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("×").on_hover_text("Clear the pick (Esc)").clicked() {
                    self.selected = None;
                }
            });
        });
        ui.label(RichText::new(&v.titles[i]).color(TEXT).strong().size(17.0));
        ui.label(RichText::new(&v.artists[i]).color(TEXT));
        ui.label(RichText::new(&v.albums[i]).color(MUTED));
        ui.add_space(6.0);
        egui::Grid::new("facts").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            let genre = v.head.genres.get(v.genre[i] as usize).cloned().unwrap_or_default();
            for (k, value) in [
                ("Genre", genre),
                ("Tempo", if v.bpm[i].is_finite() { format!("{:.0} BPM", v.bpm[i]) } else { "none found".into() }),
                ("Key", colour::key_label(&v, i)),
                ("Id", v.ids[i].to_string()),
            ] {
                ui.label(RichText::new(k).color(MUTED));
                ui.label(RichText::new(value).color(TEXT));
                ui.end_row();
            }
        });
        if ui.button("Centre on it").clicked()
            && let Some(p) = self.positions.get(i).filter(|p| p[0].is_finite()) {
                self.camera.target = glam::Vec3::from(*p);
            }

        self.for_mixing(ui, &v, i);
        if let (Some(placement), Some(&k)) = (&self.placement, self.path_of.get(&i)) {
            let phrases = placement.tracks[k].1.len();
            let shown = if self.show_paths { "drawn on the map, start to end" } else { "turn on phrase paths (P) to see them" };
            ui.label(RichText::new(format!("{phrases} phrases, {shown}.")).color(MUTED).small());
        }

        if let Some(mood) = v.head.blocks.iter().find(|b| b.name == "mood") {
            ui.separator();
            section(ui, "Mood");
            ui.label(RichText::new("Relative to the rest of the space, normalised per track.").color(MUTED).small());
            for (k, name) in mood.columns.iter().enumerate() {
                let value = unquantise(v.row(i)[mood.start + k]);
                ui.horizontal(|ui| {
                    ui.add_sized([96.0, 16.0], egui::Label::new(RichText::new(name).color(MUTED).small()));
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().min(160.0), 8.0), Sense::hover());
                    let painter = ui.painter();
                    painter.rect_filled(rect, 3.0, Color32::from_rgb(0x23, 0x26, 0x2f));
                    let mid = rect.center().x;
                    let end = mid + value.clamp(-1.0, 1.0) * rect.width() / 2.0;
                    painter.rect_filled(Rect::from_x_y_ranges(mid.min(end)..=mid.max(end), rect.y_range()), 3.0, ACCENT);
                    painter.line_segment([Pos2::new(mid, rect.top() - 2.0), Pos2::new(mid, rect.bottom() + 2.0)], Stroke::new(1.0, MUTED));
                });
            }
        }

        ui.separator();
        section(ui, "Nearest in the space");
        ui.label(RichText::new("By the weights on the left: what Stelly's neighbours would play.").color(MUTED).small());
        let mut pick = None;
        for &(j, similarity) in &self.neighbours {
            let text = format!("{} · {}", v.titles[j], v.artists[j]);
            let response = ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{similarity:.2}")).color(MUTED).small().monospace());
                ui.selectable_label(false, RichText::new(text).color(TEXT))
            });
            if response.inner.clicked() {
                pick = Some(j);
            }
        }
        if pick.is_some() {
            self.select(pick);
        }
    }

    /// The picked track as stelly-mix knows it: its whole-track tempo and
    /// key, its best mixes, a set from it, or the way to have it analysed.
    fn for_mixing(&mut self, ui: &mut egui::Ui, v: &Vectors, i: usize) {
        let id = v.ids[i];
        let ctx = ui.ctx().clone();
        ui.separator();
        section(ui, "For mixing");
        if self.mix.is_none() {
            let text = self.mix_failure.as_deref().map_or("Reading the mix library…".to_string(), |f| format!("No mix data: {f}"));
            ui.label(RichText::new(text).color(MUTED).small());
            return;
        }
        let Some(summary) = self.summary_of(id).cloned() else {
            let state = match &self.mix_state {
                Some(MixState::Queued { reason, ahead }) => format!("Queued for its analysis ({reason}), {ahead} ahead."),
                Some(MixState::Analysing) => "Being analysed now.".to_string(),
                Some(MixState::Failed { reason }) => format!("Its analysis failed: {reason}"),
                Some(MixState::Analysed { .. }) => "Analysed; reload to see it.".to_string(),
                Some(MixState::Unasked) | None => "Not analysed for mixing yet.".to_string(),
            };
            ui.label(RichText::new(state).color(TEXT));
            if self.analysing == Some(id) {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("Downloading and analysing; minutes if the server is busy.").color(MUTED).small());
                });
            } else if ui.add_enabled(self.analysing.is_none(), egui::Button::new("Analyse for mixing now")).clicked() {
                self.analysing = Some(id);
                self.analyse_failure = None;
                self.ask(&ctx, move |s| MixReply::Analysed(id, mix::analyse_now(s, id)));
            }
            if let Some(failure) = &self.analyse_failure {
                ui.label(RichText::new(failure).color(Color32::from_rgb(0xf7, 0x76, 0x8e)).small());
            }
            return;
        };

        egui::Grid::new("mix-facts").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            let excerpt = if v.bpm[i].is_finite() { format!(" (excerpt {:.0})", v.bpm[i]) } else { String::new() };
            let tempo = summary.bpm.map_or("none found".to_string(), |b| format!("{b:.1} BPM{excerpt}"));
            let excerpt_key = v.key(i).map(|k| {
                let (n, l) = k.camelot();
                format!("{n}{l}")
            });
            let key = match (&summary.key, excerpt_key) {
                (Some(whole), Some(part)) if *whole != part => format!("{whole} (excerpt {part})"),
                (Some(whole), _) => whole.clone(),
                (None, _) => "none found".into(),
            };
            let steady = if summary.steady { "steady: a set can use it" } else { "drifts: sets leave it out" };
            let length = format!("{}:{:02}, {} phrases", summary.seconds as u32 / 60, summary.seconds as u32 % 60, summary.phrases);
            for (k, value) in [("Tempo", tempo), ("Key", key), ("Beat", steady.to_string()), ("Length", length)] {
                ui.label(RichText::new(k).color(MUTED));
                ui.label(RichText::new(value).color(TEXT));
                ui.end_row();
            }
        });

        ui.add_space(6.0);
        ui.label(RichText::new("Best mixes from here").color(TEXT).strong());
        let mut pick = None;
        match &self.ways {
            None => {
                ui.spinner();
            }
            Some(Err(err)) => {
                ui.label(RichText::new(format!("{err:#}")).color(MUTED).small());
            }
            Some(Ok(ways)) if ways.is_empty() => {
                ui.label(RichText::new("Nothing in the mix library mixes with it.").color(MUTED).small());
            }
            Some(Ok(ways)) => {
                let rows = self.data.as_ref().map(|d| &d.row_of);
                for way in ways {
                    let Some(&j) = rows.and_then(|r| r.get(&way.to)) else { continue };
                    let response = ui
                        .vertical(|ui| {
                            let r = ui.horizontal(|ui| {
                                ui.label(RichText::new(format!("{:.2}", way.score)).color(MUTED).small().monospace());
                                ui.selectable_label(false, RichText::new(format!("{} · {}", v.titles[j], v.artists[j])).color(TEXT))
                            });
                            ui.label(RichText::new(way_detail(way)).color(MUTED).small());
                            r.inner
                        })
                        .inner;
                    if response.clicked() {
                        pick = Some(j);
                    }
                }
            }
        }

        ui.add_space(6.0);
        ui.label(RichText::new("A set from here").color(TEXT).strong());
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.set_minutes, 10.0..=120.0).step_by(5.0).suffix(" min"));
            if let Some(since) = self.planning {
                ui.spinner();
                ui.label(RichText::new(format!("{:.0} s", since.elapsed().as_secs_f32())).color(MUTED).small());
            } else if ui.button("Plan").on_hover_text("stelly-mix's planner, on the server").clicked() {
                self.planning = Some(Instant::now());
                self.set = None;
                let minutes = self.set_minutes;
                self.ask(&ctx, move |s| MixReply::Planned(id, mix::plan(s, id, minutes)));
            }
            if self.set.is_some() && ui.small_button("Clear").clicked() {
                self.set = None;
            }
        });
        if self.planning.is_some() {
            ui.label(RichText::new("The server plans it: about half a minute for twenty minutes of set, more for longer.").color(MUTED).small());
        }
        if !summary.steady {
            ui.label(RichText::new("Its tempo drifts, so it may not open a set: the planner may refuse it.").color(MUTED).small());
        }
        match &self.set {
            Some(Err(err)) => {
                ui.label(RichText::new(format!("{err:#}")).color(MUTED).small());
            }
            Some(Ok(plan)) => {
                ui.label(
                    RichText::new(format!("{} tracks, {:.0} min, mean mix {:.2}; drawn on the map", plan.tracks.len(), plan.seconds / 60.0, plan.mean_score))
                        .color(MUTED)
                        .small(),
                );
                let rows = self.data.as_ref().map(|d| &d.row_of);
                for (k, id) in plan.tracks.iter().enumerate() {
                    let Some(&j) = rows.and_then(|r| r.get(id)) else { continue };
                    let score = if k == 0 { "start".to_string() } else { format!("{:.2}", plan.ways[k - 1].score) };
                    let response = ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{:>2}", k + 1)).color(MUTED).small().monospace());
                        ui.label(RichText::new(score).color(MUTED).small().monospace());
                        ui.selectable_label(false, RichText::new(format!("{} · {}", v.titles[j], v.artists[j])).color(TEXT))
                    });
                    if response.inner.clicked() {
                        // Looked at, not picked: picking would drop the set.
                        if let Some(p) = self.positions.get(j).filter(|p| p[0].is_finite()) {
                            self.camera.target = glam::Vec3::from(*p);
                        }
                    }
                }
            }
            None => {}
        }
        if pick.is_some() {
            self.select(pick);
        }
    }

    fn status(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let Some(data) = &self.data else {
                ui.label(RichText::new("No space yet").color(MUTED));
                return;
            };
            let shown = self.positions.iter().filter(|p| p[0].is_finite()).count();
            ui.label(RichText::new(format!("{shown} of {} tracks", data.v.head.n)).color(MUTED));
            ui.separator();
            ui.label(RichText::new(self.projection.describe(&data.v)).color(MUTED));
            if let Some(took) = self.took {
                ui.label(RichText::new(format!("({took:.1} s)")).color(MUTED));
            }
            if let Some(job) = &self.job {
                ui.separator();
                ui.label(RichText::new(format!("computing {}…", job.projection.describe(&data.v))).color(ACCENT));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mean = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
                ui.label(RichText::new(format!("{:.0} fps", 1.0 / mean.max(1e-3))).color(MUTED).small());
                let hint = if self.three() { "drag: turn · right-drag or shift: move · scroll: zoom · F: fit" } else { "drag: move · scroll: zoom · F: fit" };
                ui.label(RichText::new(hint).color(MUTED).small());
            });
        });
    }
}

// -------------------------------------------------------------------- view

impl Explorer {
    fn view(&mut self, ui: &mut egui::Ui) {
        self.rebuild();
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let three = self.three();
        let aspect = rect.width() / rect.height().max(1.0);
        let ctx = ui.ctx().clone();

        // ------------------------------------------------------ input
        if response.dragged_by(egui::PointerButton::Primary) && !ctx.input(|i| i.modifiers.shift) {
            let d = response.drag_delta();
            self.camera.drag(three, glam::Vec2::new(d.x, d.y), rect.height());
        } else if response.dragged_by(egui::PointerButton::Secondary) || response.dragged_by(egui::PointerButton::Middle) || (response.dragged() && ctx.input(|i| i.modifiers.shift)) {
            let d = response.drag_delta();
            self.camera.pan(three, glam::Vec2::new(d.x, d.y), rect.height());
        }
        if let Some(pointer) = response.hover_pos() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            let pinch = ctx.input(|i| i.zoom_delta());
            let factor = (scroll * 0.0025).exp() * pinch;
            if (factor - 1.0).abs() > 1e-4 {
                let at = glam::Vec2::new((pointer.x - rect.center().x) / (rect.width() / 2.0), -(pointer.y - rect.center().y) / (rect.height() / 2.0));
                self.camera.zoom(three, factor, at, aspect);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F)) && !ctx.egui_wants_keyboard_input() {
            self.camera = Camera::default();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.select(None);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::P)) && !ctx.egui_wants_keyboard_input() {
            self.show_paths = !self.show_paths;
            if self.show_paths {
                self.paths_on(&ctx);
            }
            self.dirty = true;
        }

        // --------------------------------------------------- the points
        let view_proj = self.camera.view_proj(three, aspect);
        let ppp = ctx.pixels_per_point();
        let draw = Draw {
            points: self.points.clone(),
            lines: self.lines.clone(),
            line_width: self.path_width * ppp,
            view_proj,
            radius: self.radius * ppp,
            viewport: [rect.width() * ppp, rect.height() * ppp],
        };
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, draw));

        // ------------------------------------------------------ picking
        let to_screen = |p: [f32; 3]| -> Option<(Pos2, f32)> {
            let c = view_proj * glam::Vec4::new(p[0], p[1], p[2], 1.0);
            if c.w <= 0.0 {
                return None;
            }
            let (x, y, z) = (c.x / c.w, c.y / c.w, c.z / c.w);
            Some((Pos2::new(rect.center().x + x * rect.width() / 2.0, rect.center().y - y * rect.height() / 2.0), z))
        };
        let key = (view_proj.to_cols_array(), rect);
        let pointer = response.hover_pos();
        if pointer.is_some() && self.screen_for != Some(key) {
            self.screen.clear();
            for &i in &self.drawn {
                if !self.lit(i as usize) {
                    continue;
                }
                if let Some((s, z)) = to_screen(self.positions[i as usize])
                    && rect.contains(s) && (0.0..=1.0).contains(&z) {
                        self.screen.push((i, s.x, s.y, z));
                    }
            }
            self.screen_for = Some(key);
        }
        self.hover = pointer.and_then(|p| {
            let reach = (REACH + self.radius).powi(2);
            self.screen
                .iter()
                .filter(|(_, x, y, _)| (x - p.x).powi(2) + (y - p.y).powi(2) <= reach)
                .min_by(|a, b| {
                    let da = (a.1 - p.x).powi(2) + (a.2 - p.y).powi(2);
                    let db = (b.1 - p.x).powi(2) + (b.2 - p.y).powi(2);
                    // Nearer to the eye first when two are about as close.
                    (da + a.3 * 40.0).total_cmp(&(db + b.3 * 40.0))
                })
                .map(|(i, ..)| *i as usize)
        });
        if response.clicked() {
            self.select(self.hover);
        }

        // ----------------------------------------------------- overlays
        let painter = ui.painter_at(rect);
        let Some(data) = &self.data else { return };
        let v = data.v.clone();

        if let Projection::Axes { columns, .. } = &self.projection {
            axes(&painter, rect, three, &to_screen, &v, columns);
        } else if three {
            for (axis, end) in [(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), (2, [0.0, 0.0, 1.0])] {
                let mut start = [0.0; 3];
                start[axis] = -1.0;
                if let (Some((a, _)), Some((b, _))) = (to_screen(start), to_screen(end)) {
                    painter.line_segment([a, b], Stroke::new(1.0, MUTED.linear_multiply(0.25)));
                }
            }
        }

        if self.show_labels {
            self.draw_labels(&painter, rect, &to_screen, &v);
        }

        // A planned set: its route, numbered, under the picked track's marks.
        if let Some(Ok(plan)) = &self.set {
            let rows = self.data.as_ref().map(|d| &d.row_of);
            let stops: Vec<Pos2> = plan
                .tracks
                .iter()
                .filter_map(|id| rows.and_then(|r| r.get(id)))
                .filter_map(|&j| self.positions.get(j).filter(|p| p[0].is_finite()).and_then(|p| to_screen(*p)))
                .map(|(s, _)| s)
                .collect();
            let [r, g, b, _] = colour::DRIFTING;
            let route = Color32::from_rgb(r, g, b);
            for pair in stops.windows(2) {
                painter.line_segment([pair[0], pair[1]], Stroke::new(2.0, route));
            }
            for (k, at) in stops.iter().enumerate() {
                painter.circle_filled(*at, 8.0, BACKGROUND);
                painter.circle_stroke(*at, 8.0, Stroke::new(1.5, route));
                painter.text(*at, Align2::CENTER_CENTER, (k + 1).to_string(), FontId::proportional(10.0), TEXT);
            }
        }

        // The picked track's own path, drawn out: its phrases, where it
        // starts and where it ends.
        if self.show_paths
            && let (Some(i), Some(placement)) = (self.selected, &self.placement)
            && let Some(&k) = self.path_of.get(&i)
        {
            let range = placement.tracks[k].1.clone();
            let at: Vec<(Pos2, f32)> = range
                .clone()
                .filter_map(|ph| {
                    let p = self.phrase_positions.get(ph).filter(|p| p[0].is_finite())?;
                    Some((to_screen(*p)?.0, placement.starts[ph]))
                })
                .collect();
            let curve: Vec<Pos2> = phrase_paths::smooth(&at.iter().map(|(p, _)| [p.x, p.y, 0.0]).collect::<Vec<_>>(), 6)
                .into_iter()
                .map(|q| Pos2::new(q[0], q[1]))
                .collect();
            painter.add(egui::Shape::line(curve.clone(), Stroke::new(4.0, BACKGROUND)));
            painter.add(egui::Shape::line(curve, Stroke::new(2.0, TEXT)));
            let clock = |s: f32| format!("{}:{:02}", s as u32 / 60, s as u32 % 60);
            for (n, (q, start)) in at.iter().enumerate() {
                painter.circle_filled(*q, 3.0, TEXT);
                if n == 0 || n + 1 == at.len() {
                    let label = if n == 0 { format!("starts {}", clock(*start)) } else { format!("last phrase {}", clock(*start)) };
                    painter.text(*q + Vec2::new(8.0, -8.0), Align2::LEFT_BOTTOM, label, FontId::proportional(11.0), TEXT);
                }
            }
        }

        // The best ways out of the picked track, as arrows.
        if self.show_ways
            && let (Some(i), Some(Ok(ways))) = (self.selected, &self.ways)
            && let Some((s, _)) = self.positions.get(i).filter(|p| p[0].is_finite()).and_then(|p| to_screen(*p))
        {
            let rows = self.data.as_ref().map(|d| &d.row_of);
            let [r, g, b, _] = colour::READY;
            let arrow = Color32::from_rgb(r, g, b);
            for way in ways {
                let Some(&j) = rows.and_then(|r| r.get(&way.to)) else { continue };
                let Some((t, _)) = self.positions.get(j).filter(|p| p[0].is_finite()).and_then(|p| to_screen(*p)) else { continue };
                let length = (t - s).length();
                if length < 1.0 {
                    continue;
                }
                let direction = (t - s) / length;
                let tip = t - direction * (self.radius + 4.0);
                painter.line_segment([s, tip], Stroke::new(1.5, arrow.linear_multiply(0.8)));
                let side = Vec2::new(-direction.y, direction.x) * 4.0;
                painter.add(egui::Shape::convex_polygon(vec![tip, tip - direction * 8.0 + side, tip - direction * 8.0 - side], arrow, Stroke::NONE));
            }
        }

        if let Some(i) = self.selected
            && let Some((s, _)) = self.positions.get(i).filter(|p| p[0].is_finite()).and_then(|p| to_screen(*p)) {
                if self.show_links {
                    for &(j, _) in &self.neighbours {
                        if let Some((t, _)) = self.positions.get(j).filter(|p| p[0].is_finite()).and_then(|p| to_screen(*p)) {
                            painter.line_segment([s, t], Stroke::new(1.0, ACCENT.linear_multiply(0.55)));
                            painter.circle_stroke(t, self.radius + 3.0, Stroke::new(1.0, ACCENT.linear_multiply(0.8)));
                        }
                    }
                }
                painter.circle_stroke(s, self.radius + 6.0, Stroke::new(3.0, BACKGROUND));
                painter.circle_stroke(s, self.radius + 5.0, Stroke::new(2.0, TEXT));
            }

        if let Some(i) = self.hover
            && let Some((s, _)) = self.positions.get(i).and_then(|p| to_screen(*p)) {
                painter.circle_stroke(s, self.radius + 4.0, Stroke::new(1.5, ACCENT));
                let genre = v.head.genres.get(v.genre[i] as usize).cloned().unwrap_or_default();
                let tempo = if v.bpm[i].is_finite() { format!("{:.0} BPM", v.bpm[i]) } else { "tempo unknown".into() };
                let key = v.key(i).map_or("key unknown".to_string(), |k| {
                    let (n, l) = k.camelot();
                    format!("{n}{l} {}", k.name())
                });
                let mut lines = vec![(v.titles[i].clone(), TEXT, true), (format!("{} · {genre}", v.artists[i]), MUTED, false), (format!("{tempo} · {key}"), MUTED, false)];
                if let Some(summary) = self.summary_of(v.ids[i]) {
                    let bpm = summary.bpm.map_or(String::new(), |b| format!(" · {b:.1} BPM"));
                    let state = if summary.steady { "ready to mix" } else { "analysed, tempo drifts" };
                    lines.push((format!("{state}{bpm}"), MUTED, false));
                }
                tooltip(&painter, rect, s, &lines);
                ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
            }

        if self.job.is_some() && self.positions.is_empty() {
            painter.text(rect.center(), Align2::CENTER_CENTER, "Projecting…", FontId::proportional(16.0), MUTED);
        }
        if response.dragged() || self.job.is_some() {
            ctx.request_repaint();
        }
    }

    fn draw_labels(&self, painter: &egui::Painter, rect: Rect, to_screen: &dyn Fn([f32; 3]) -> Option<(Pos2, f32)>, v: &Vectors) {
        let genre_colour = |g: u16| -> Color32 {
            if let Legend::Categories(rows) = &self.legend {
                let name = v.head.genres.get(g as usize);
                if let Some((_, [r, gg, b, _], _)) = rows.iter().find(|(n, ..)| Some(n) == name) {
                    return Color32::from_rgb(*r, *gg, *b);
                }
            }
            MUTED
        };
        let mut taken: Vec<(Rect, u16, Pos2)> = Vec::new();
        let font = FontId::proportional(12.5);
        for region in &self.regions {
            if taken.len() >= LABELS {
                break;
            }
            let Some((at, z)) = to_screen(region.position) else { continue };
            if !(0.0..=1.0).contains(&z) {
                continue;
            }
            let name = v.head.genres.get(region.genre as usize).cloned().unwrap_or_default();
            let galley = painter.layout_no_wrap(name, font.clone(), TEXT);
            let size = galley.size() + Vec2::new(22.0, 6.0);
            let chip = Rect::from_center_size(at, size);
            if !rect.shrink(4.0).contains_rect(chip) {
                continue;
            }
            if taken.iter().any(|(t, g, p)| t.expand2(Vec2::new(8.0, 4.0)).intersects(chip) || (*g == region.genre && p.distance(at) < 160.0)) {
                continue;
            }
            taken.push((chip, region.genre, at));
            painter.rect_filled(chip, 6.0, BACKGROUND.linear_multiply(0.82));
            painter.circle_filled(Pos2::new(chip.left() + 9.0, at.y), 3.5, genre_colour(region.genre));
            painter.galley(Pos2::new(chip.left() + 17.0, chip.top() + 3.0), galley, TEXT);
        }
    }
}

/// Whether the picked track is another than the one asked about, which then
/// becomes it.
fn picked_changed(picked: Option<i64>, asked_for: &mut Option<i64>) -> bool {
    if picked == *asked_for {
        return false;
    }
    *asked_for = picked;
    true
}

/// A way's tempo, key and where it happens, as a line under its track.
fn way_detail(way: &Way) -> String {
    let clock = |s: f64| format!("{}:{:02}", s as u32 / 60, s as u32 % 60);
    let keys = match way.key_steps {
        Some(0) => " · same key".to_string(),
        Some(1) => " · 1 key step".to_string(),
        Some(n) => format!(" · {n} key steps"),
        None => String::new(),
    };
    format!("{:+.1}% tempo{keys} · sound {:.2} · out at {} → in at {}", (way.ratio - 1.0) * 100.0, way.sound, clock(way.out_at.0), clock(way.into_at.0))
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title.to_uppercase()).color(MUTED).small().strong());
}

fn swatch(ui: &mut egui::Ui, [r, g, b, _]: Rgba) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 5.0, Color32::from_rgb(r, g, b));
}

fn column_picker(ui: &mut egui::Ui, v: &Vectors, label: &str, value: &mut Option<usize>, optional: bool) {
    let text = value.map_or("none".to_string(), |c| project::column_name(v, c));
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(MUTED));
        egui::ComboBox::from_id_salt(label).selected_text(text).width(200.0).show_ui(ui, |ui| {
            if optional {
                ui.selectable_value(value, None, "none");
            }
            for b in &v.head.blocks {
                // The semantic and style blocks are PCA components with no
                // meaning of their own; listed last, under their numbers.
                for (k, name) in b.columns.iter().enumerate() {
                    ui.selectable_value(value, Some(b.start + k), format!("{} · {name}", b.name));
                }
            }
        });
    });
}

fn axes(painter: &egui::Painter, rect: Rect, three: bool, to_screen: &dyn Fn([f32; 3]) -> Option<(Pos2, f32)>, v: &Vectors, columns: &[Option<usize>; 3]) {
    let stroke = Stroke::new(1.0, MUTED.linear_multiply(0.35));
    let font = FontId::proportional(12.0);
    for axis in 0..(if three { 3 } else { 2 }) {
        let Some(c) = columns[axis] else { continue };
        let (mut a, mut b) = ([0f32; 3], [0f32; 3]);
        a[axis] = -1.0;
        b[axis] = 1.0;
        if !three {
            // Along the bottom and the left edge, where a chart's axes go.
            a[1 - axis] = -1.05;
            b[1 - axis] = -1.05;
        }
        let (Some((sa, _)), Some((sb, _))) = (to_screen(a), to_screen(b)) else { continue };
        painter.line_segment([sa, sb], stroke);
        let name = project::column_name(v, c);
        let at = Pos2::new(sb.x.clamp(rect.left() + 60.0, rect.right() - 60.0), sb.y.clamp(rect.top() + 12.0, rect.bottom() - 12.0));
        painter.text(at, Align2::CENTER_CENTER, format!("{name} →"), font.clone(), MUTED);
    }
}

fn tooltip(painter: &egui::Painter, rect: Rect, at: Pos2, lines: &[(String, Color32, bool)]) {
    let galleys: Vec<_> = lines
        .iter()
        .map(|(text, colour, strong)| painter.layout_no_wrap(text.clone(), if *strong { FontId::proportional(13.5) } else { FontId::proportional(12.0) }, *colour))
        .collect();
    let width = galleys.iter().map(|g| g.size().x).fold(0.0, f32::max) + 16.0;
    let height = galleys.iter().map(|g| g.size().y + 2.0).sum::<f32>() + 10.0;
    let mut origin = at + Vec2::new(14.0, -height / 2.0);
    if origin.x + width > rect.right() - 4.0 {
        origin.x = at.x - width - 14.0;
    }
    origin.y = origin.y.clamp(rect.top() + 4.0, rect.bottom() - height - 4.0);
    let frame = Rect::from_min_size(origin, Vec2::new(width, height));
    painter.rect_filled(frame, 7.0, Color32::from_rgb(0x23, 0x26, 0x2f));
    painter.rect_stroke(frame, 7.0, Stroke::new(1.0, Color32::from_rgb(0x2c, 0x30, 0x3b)), egui::StrokeKind::Inside);
    let mut y = origin.y + 5.0;
    for (galley, (_, colour, _)) in galleys.into_iter().zip(lines) {
        let h = galley.size().y;
        painter.galley(Pos2::new(origin.x + 8.0, y), galley, *colour);
        y += h + 2.0;
    }
}
