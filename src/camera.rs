//! Looking at the projection: flat and orthographic in 2D, orbiting with
//! perspective in 3D. Projections fit within -1 and 1, so both frame that.
//! wgpu's depth runs 0 to 1 with y up, glam's `directx` convention.

use glam::camera::rh::{proj::directx, view::look_at_mat4};
use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// What is looked at: the middle of the view in 2D, the orbit's pivot in 3D.
    pub target: Vec3,
    /// 2D: half the view's height, in projection units.
    pub half_height: f32,
    /// 3D: radians round the vertical, and up from the horizontal.
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

const FOV: f32 = 0.8;

impl Default for Camera {
    fn default() -> Self {
        Self { target: Vec3::ZERO, half_height: 1.15, yaw: 0.6, pitch: 0.45, distance: 4.8 }
    }
}

impl Camera {
    pub fn view_proj(&self, three: bool, aspect: f32) -> Mat4 {
        if three {
            let eye = self.target + self.direction() * self.distance;
            let near = (self.distance * 0.01).max(1e-4);
            directx::perspective(FOV, aspect, near, self.distance + 10.0) * look_at_mat4(eye, self.target, Vec3::Y)
        } else {
            let (h, w) = (self.half_height, self.half_height * aspect);
            let t = self.target;
            directx::orthographic(t.x - w, t.x + w, t.y - h, t.y + h, -10.0, 10.0)
        }
    }

    /// From the target towards the eye.
    fn direction(&self) -> Vec3 {
        Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos())
    }

    /// A drag of `delta` pixels in a view `height` pixels tall: the plane
    /// follows the pointer in 2D, the view turns in 3D.
    pub fn drag(&mut self, three: bool, delta: Vec2, height: f32) {
        if three {
            self.yaw -= delta.x * 0.008;
            self.pitch = (self.pitch + delta.y * 0.008).clamp(-1.55, 1.55);
        } else {
            self.pan(false, delta, height);
        }
    }

    /// Moves what is looked at with the pointer, in both modes.
    pub fn pan(&mut self, three: bool, delta: Vec2, height: f32) {
        let per_pixel = if three { 2.0 * self.distance * (FOV / 2.0).tan() / height } else { 2.0 * self.half_height / height };
        if three {
            let forward = -self.direction();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(forward);
            self.target += (-right * delta.x + up * delta.y) * per_pixel;
        } else {
            self.target.x -= delta.x * per_pixel;
            self.target.y += delta.y * per_pixel;
        }
    }

    /// Zoom by `factor` (above 1 is in), keeping the point under the pointer
    /// put: `at` is its place in the view, -1 to 1 on each axis, y up.
    pub fn zoom(&mut self, three: bool, factor: f32, at: Vec2, aspect: f32) {
        if three {
            let before = self.distance;
            self.distance = (self.distance / factor).clamp(0.05, 50.0);
            // Toward what is under the pointer, as a map does.
            let shift = before - self.distance;
            let forward = -self.direction();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(forward);
            let half = (FOV / 2.0).tan();
            self.target += (right * at.x * half * aspect + up * at.y * half) * shift;
        } else {
            let under = self.target.truncate() + Vec2::new(at.x * self.half_height * aspect, at.y * self.half_height);
            self.half_height = (self.half_height / factor).clamp(1e-4, 50.0);
            let after = self.target.truncate() + Vec2::new(at.x * self.half_height * aspect, at.y * self.half_height);
            self.target += (under - after).extend(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen(m: Mat4, p: Vec3) -> Vec2 {
        let c = m * p.extend(1.0);
        Vec2::new(c.x / c.w, c.y / c.w)
    }

    #[test]
    fn zooming_keeps_the_point_under_the_pointer() {
        let mut camera = Camera::default();
        let aspect = 1.5;
        let at = Vec2::new(0.4, -0.3);
        let m = camera.view_proj(false, aspect);
        let world = m.inverse().transform_point3(at.extend(0.0));
        camera.zoom(false, 2.0, at, aspect);
        let back = screen(camera.view_proj(false, aspect), world);
        assert!((back - at).length() < 1e-4, "{back}");
    }

    #[test]
    fn the_default_view_frames_the_unit_box() {
        let camera = Camera::default();
        for three in [false, true] {
            let m = camera.view_proj(three, 1.0);
            for corner in [Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 1.0)] {
                let p = screen(m, if three { corner } else { corner.truncate().extend(0.0) });
                assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0, "{three} {corner} -> {p}");
            }
        }
    }
}
