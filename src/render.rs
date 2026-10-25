//! The points, drawn by the GPU inside egui's own render pass: each one a
//! small square facing the screen, cut to an antialiased disc in the
//! fragment shader. Positions and colours go up once per change, as one
//! instance buffer; a frame only sends the camera.
//!
//! Depth-tested, so in 3D the near points cover the far ones whatever order
//! they come in; `LessEqual`, so in 2D, where every point is at one depth,
//! the later ones still draw. A point faded below half opacity, dimmed by a
//! search or a focus, is pushed to the back, so it never hides a lit one.

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use std::sync::Arc;

/// One point as the GPU reads it: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub position: [f32; 3],
    pub colour: [u8; 4],
}

/// What is drawn, shared with the callback by `Arc` and replaced whole when
/// it changes; `version` is how the GPU side knows.
pub struct Points {
    pub version: u64,
    pub instances: Vec<Instance>,
}

/// One straight piece of a curve: 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Line {
    pub from: [f32; 3],
    pub to: [f32; 3],
    pub colour: [u8; 4],
}

/// The lines drawn under the points, as `Points` is.
pub struct Lines {
    pub version: u64,
    pub lines: Vec<Line>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    radius: f32,
    line_width: f32,
}

pub const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
    radius: f32,
    line_width: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct LineOut {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec4<f32>,
    @location(1) across: f32,
};

// A line piece as a quad, as wide on screen as `line_width` pixels plus one
// for its soft edge, whatever its depth.
@vertex
fn line_vs(@builtin(vertex_index) vertex: u32, @location(0) head: vec3<f32>, @location(1) tail: vec3<f32>, @location(2) colour: vec4<f32>) -> LineOut {
    var ends = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var sides = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let a = u.view_proj * vec4(head, 1.0);
    let b = u.view_proj * vec4(tail, 1.0);
    let pa = a.xy / a.w * u.viewport;
    let pb = b.xy / b.w * u.viewport;
    var along = pb - pa;
    if (length(along) < 1e-6) {
        along = vec2(1.0, 0.0);
    }
    let normal = normalize(vec2(-along.y, along.x));
    let reach = (u.line_width + 1.0) * 0.5;
    let end = select(a, b, ends[vertex] > 0.5);
    var out: LineOut;
    out.position = vec4(end.xy + normal * sides[vertex] * reach * 2.0 / u.viewport * end.w, end.zw);
    out.colour = colour;
    out.across = sides[vertex] * reach;
    return out;
}

@fragment
fn line_fs(in: LineOut) -> @location(0) vec4<f32> {
    let coverage = clamp(u.line_width * 0.5 + 0.5 - abs(in.across), 0.0, 1.0);
    return vec4(in.colour.rgb, in.colour.a * coverage);
}

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec4<f32>,
    @location(1) corner: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @location(0) position: vec3<f32>, @location(1) colour: vec4<f32>) -> Out {
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let corner = corners[vertex];
    // A pixel or more of square past the disc, for its soft edge.
    let reach = u.radius + 1.0;
    var clip = u.view_proj * vec4(position, 1.0);
    clip = vec4(clip.xy + corner * reach * 2.0 / u.viewport * clip.w, clip.zw);
    var out: Out;
    out.position = clip;
    out.colour = colour;
    out.corner = corner * reach;
    return out;
}

struct Fragment {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: Out) -> Fragment {
    // Distance from the centre in pixels, against the radius: a disc with
    // one pixel of edge, and a point under a pixel wide still lands as a
    // faint pixel rather than nothing.
    let distance = length(in.corner);
    let radius = max(u.radius, 0.5);
    let coverage = clamp(radius + 0.5 - distance, 0.0, 1.0) * min(1.0, u.radius / 0.5);
    if coverage <= 0.0 {
        discard;
    }
    var out: Fragment;
    out.colour = vec4(in.colour.rgb, in.colour.a * coverage);
    out.depth = select(in.position.z, 1.0, in.colour.a < 0.45);
    return out;
}
"#;

/// Lives in egui's callback resources: made once, kept for the app's life.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    lines: wgpu::Buffer,
    line_capacity: usize,
    line_count: u32,
    line_version: u64,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    instances: wgpu::Buffer,
    capacity: usize,
    count: u32,
    version: u64,
}

impl Renderer {
    pub fn install(state: &egui_wgpu::RenderState) {
        let renderer = Self::new(&state.device, state.target_format);
        state.renderer.write().callback_resources.insert(renderer);
    }

    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("points"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        // The lines: no depth written, so the points drawn after them stay
        // on top, but tested, so in 3D a near cloud hides a far path.
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("line_vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Line>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("line_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("points"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let instances = Self::buffer(device, 1);
        let lines = Self::line_buffer(device, 1);
        Renderer {
            pipeline,
            line_pipeline,
            lines,
            line_capacity: 1,
            line_count: 0,
            line_version: u64::MAX,
            bind_group,
            uniforms,
            instances,
            capacity: 1,
            count: 0,
            version: u64::MAX,
        }
    }

    fn line_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lines"),
            size: (capacity * std::mem::size_of::<Line>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points instances"),
            size: (capacity * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// One frame's draw: the points as they stand and where the camera is.
pub struct Draw {
    pub points: Arc<Points>,
    pub lines: Arc<Lines>,
    /// Physical pixels.
    pub line_width: f32,
    pub view_proj: glam::Mat4,
    /// Radius in physical pixels.
    pub radius: f32,
    /// The rect drawn into, in physical pixels.
    pub viewport: [f32; 2],
}

impl egui_wgpu::CallbackTrait for Draw {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<Renderer>() else { return Vec::new() };
        if r.version != self.points.version {
            let needed = self.points.instances.len().max(1);
            if needed > r.capacity {
                // Room to grow, so a slightly bigger set does not reallocate.
                r.capacity = needed + needed / 4;
                r.instances = Renderer::buffer(device, r.capacity);
            }
            queue.write_buffer(&r.instances, 0, bytemuck::cast_slice(&self.points.instances));
            r.count = self.points.instances.len() as u32;
            r.version = self.points.version;
        }
        if r.line_version != self.lines.version {
            let needed = self.lines.lines.len().max(1);
            if needed > r.line_capacity {
                r.line_capacity = needed + needed / 4;
                r.lines = Renderer::line_buffer(device, r.line_capacity);
            }
            queue.write_buffer(&r.lines, 0, bytemuck::cast_slice(&self.lines.lines));
            r.line_count = self.lines.lines.len() as u32;
            r.line_version = self.lines.version;
        }
        let _ = screen;
        let uniforms = Uniforms {
            view_proj: self.view_proj.to_cols_array_2d(),
            viewport: [self.viewport[0].max(1.0), self.viewport[1].max(1.0)],
            radius: self.radius,
            line_width: self.line_width,
        };
        queue.write_buffer(&r.uniforms, 0, bytemuck::bytes_of(&uniforms));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        let Some(r) = resources.get::<Renderer>() else { return };
        pass.set_bind_group(0, &r.bind_group, &[]);
        if r.line_count > 0 {
            pass.set_pipeline(&r.line_pipeline);
            pass.set_vertex_buffer(0, r.lines.slice(..(r.line_count as u64 * std::mem::size_of::<Line>() as u64)));
            pass.draw(0..6, 0..r.line_count);
        }
        if r.count == 0 {
            return;
        }
        pass.set_pipeline(&r.pipeline);
        pass.set_vertex_buffer(0, r.instances.slice(..(r.count as u64 * std::mem::size_of::<Instance>() as u64)));
        pass.draw(0..6, 0..r.count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui_wgpu::CallbackTrait;

    const SIZE: u32 = 64;

    /// The real pipeline, drawn off screen: a red point in front of a blue
    /// one at the centre, a green one in a corner, and a dimmed one that
    /// must not cover anything. Skipped where there is no adapter at all.
    #[test]
    fn draws_points_where_the_camera_puts_them() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("no wgpu adapter here; skipping");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(Renderer::new(&device, format));

        let camera = crate::camera::Camera::default();
        let draw = Draw {
            points: Arc::new(Points {
                version: 1,
                instances: vec![
                    // Behind, then in front, then a dimmed one nearer still.
                    Instance { position: [0.0, 0.0, -0.5], colour: [0, 0, 255, 255] },
                    Instance { position: [0.0, 0.0, 0.5], colour: [255, 0, 0, 255] },
                    Instance { position: [0.0, 0.0, 0.9], colour: [0, 255, 0, 20] },
                    Instance { position: [0.8, 0.8, 0.0], colour: [0, 255, 0, 255] },
                ],
            }),
            // And a line, away from the points.
            lines: Arc::new(Lines { version: 1, lines: vec![Line { from: [-0.6, -0.6, 0.0], to: [-0.6, 0.3, 0.0], colour: [255, 255, 255, 255] }] }),
            line_width: 3.0,
            view_proj: camera.view_proj(true, 1.0),
            radius: 4.0,
            viewport: [SIZE as f32; 2],
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [SIZE, SIZE], pixels_per_point: 1.0 };
        draw.prepare(&device, &queue, &screen, &mut encoder, &mut resources);
        {
            let view = target.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let mut pass = pass.forget_lifetime();
            let info = egui::PaintCallbackInfo {
                viewport: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(SIZE as f32)),
                clip_rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(SIZE as f32)),
                pixels_per_point: 1.0,
                screen_size_px: [SIZE, SIZE],
            };
            draw.paint(info, &mut pass, &resources);
        }
        let row = (SIZE * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * SIZE) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
            wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap().to_vec();
        let pixel = |p: [f32; 3]| {
            let c = camera.view_proj(true, 1.0) * glam::Vec4::new(p[0], p[1], p[2], 1.0);
            let x = ((c.x / c.w + 1.0) / 2.0 * SIZE as f32) as u32;
            let y = ((1.0 - c.y / c.w) / 2.0 * SIZE as f32) as u32;
            let at = (y * row + x * 4) as usize;
            [bytes[at], bytes[at + 1], bytes[at + 2]]
        };
        let centre = pixel([0.0, 0.0, 0.5]);
        assert!(centre[0] > 200 && centre[1] < 60 && centre[2] < 60, "the near red point should cover the blue and the dimmed: {centre:?}");
        let corner = pixel([0.8, 0.8, 0.0]);
        assert!(corner[1] > 200 && corner[0] < 60, "a green point where the camera puts it: {corner:?}");
        let along = pixel([-0.6, -0.15, 0.0]);
        assert!(along.iter().all(|c| *c > 200), "the line, drawn where the camera puts it: {along:?}");
        assert_eq!(&bytes[0..3], &[0, 0, 0], "nothing drawn elsewhere");
    }
}
