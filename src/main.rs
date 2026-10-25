//! Stelly Explorer: a Stelly server's space, drawn whole, in the server's own
//! map or in projections made here, flat or in 3D, coloured by genre, key,
//! tempo or any column of the space.

mod app;
mod camera;
mod colour;
mod labels;
mod mix;
mod project;
mod render;
mod source;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Stelly Explorer")
            .with_app_id("stelly-explorer")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([720.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        // The points are depth-tested in 3D; see `render`.
        depth_buffer: 24,
        ..Default::default()
    };
    eframe::run_native("stelly-explorer", options, Box::new(|cc| Ok(Box::new(app::Explorer::new(cc)))))
}
