extern crate nalgebra as na;

type Color = na::Vector3<u8>;
type Vec2 = na::Vector2<f32>;
type Vec3 = na::Vector3<f32>;
struct Tri(Vec3, Vec3, Vec3);

#[derive(Debug)]
struct Line2(Vec2, Vec2);
#[derive(Debug)]
struct Line3(Vec3, Vec3);

macro_rules! convpt {
    ($p:expr => Vec2) => {
        Vec2::new($p.x, $p.y)
    };
    ($p:expr => Vec3) => {
        Vec3::new($p.x, $p.y, $p.z)
    };
}

macro_rules! pt {
    ($x:expr, $y:expr) => {
        Vec2::new($x, $y)
    };
    ($x:expr, $y:expr, $z:expr) => {
        Vec3::new($x, $y, $z)
    };
}

#[derive(Debug)]
struct RenderState {
    n: u64,
    t: f32,
    last_frame_start: std::time::Instant,
}

#[derive(Debug)]
struct Camera {
    pos: Vec3,
    rot: na::Rotation3<f32>,
    pub near_clip_plane: f32,
    pub hfov: f32,
}

impl Camera {
    // Returns the world to camera transformation matrix
    fn transform_matrix(&self) -> na::Matrix4<f32> {
        let translation = na::Translation3::from(self.pos).inverse();
        translation.to_homogeneous() * self.rot.to_homogeneous()
    }
}

#[derive(Debug)]
pub struct Renderer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    state: RenderState,
    camera: Camera,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Renderer {
        Renderer {
            width,
            height,
            buffer: vec![0; width * height],
            state: RenderState {
                t: 0.0,
                n: 0,
                last_frame_start: std::time::Instant::now(),
            },
            camera: Camera {
                pos: pt!(0.0, 0.0, 5.0), // In front of XY plane
                rot: na::Rotation3::from_axis_angle(&Vec3::x_axis(), std::f32::consts::FRAC_PI_2), // facing negative z
                near_clip_plane: 0.1,
                hfov: 1.1,
            },
        }
    }
    pub fn draw(&mut self) -> Result<(), String> {
        let dt = self.state.last_frame_start.elapsed().as_secs_f32();
        self.state.t += dt;
        let n = self.state.n;

        if n % 120 == 0 {
            println!(
                "draw: {:?} ({} fps)",
                self.state.last_frame_start.elapsed(),
                1.0 / (dt)
            );
        }
        self.state.last_frame_start = std::time::Instant::now();

        for i in self.buffer.iter_mut() {
            // Clear buffer
            *i = 0x00000000; // __RRGGBB
        }

        let rot = na::Rotation3::from_axis_angle(&Vec3::x_axis(), self.state.t)
            * na::Rotation3::from_axis_angle(&Vec3::y_axis(), self.state.t / 2.0)
            * na::Rotation3::from_axis_angle(&Vec3::z_axis(), self.state.t / 1.5);

        let a = rot * pt!(0.0, 0.0, 0.0);
        let b = rot * pt!(1.0, 0.0, 0.0);
        let c = rot * pt!(0.5, 0.0, 1.0);
        let d = rot * pt!(0.5, 1.0, 0.5);

        let tetra: Vec<Tri> = vec![Tri(a, b, c), Tri(a, b, d), Tri(a, d, c), Tri(d, b, c)];
        self.wireframe(&tetra, Color::new(0, 255, 0));

        self.line(
            &Line2(
                pt!(300.0, 300.0),
                pt!(
                    300.0 + 300.0 * (self.state.t / 10.0).cos(),
                    300.0 + 300.0 * (self.state.t / 10.0).sin()
                ),
            ),
            Color::new(255, 255, 0),
        );
        self.line(
            &Line2(
                pt!(900.0, 900.0),
                pt!(
                    300.0 - 300.0 * (self.state.t / 10.0).cos(),
                    300.0 + 300.0 * (self.state.t / 10.0).sin()
                ),
            ),
            Color::new(255, 255, 0),
        );
        self.state.n += 1;
        Ok(())
    }

    // Plots a single pixel with the given color
    fn plot(&mut self, pos: Vec2, col: Color) {
        let index = pos.y as usize * self.width + pos.x as usize;
        let pixel = &mut self.buffer[index];
        *pixel = (col.x as u32) << 16 as u32;
        *pixel |= (col.y as u32) << 8 as u32;
        *pixel |= col.z as u32;
    }

    fn line(&mut self, line: &Line2, col: Color) {
        // If either vertex is off screen, move it back on screen
        let clip = |p: Vec2, o: Vec2| -> Vec2 {
            let mut new_x = p.x;
            let mut new_y = p.y;
            let x_slope = (p.y - o.y) / (p.x - o.x);
            let y_slope = (p.x - o.x) / (p.y - o.y);
            let x_max = self.width as f32 - 1.0;
            let y_max = self.height as f32 - 1.0;
            let (x_min, y_min) = (1.0, 1.0);

            if new_y > y_max {
                new_x += y_slope * (y_max - new_y);
                new_y = y_max;
            } else if new_y < y_min {
                new_x += y_slope * (y_min - new_y);
                new_y = y_min;
            }
            if new_x > x_max {
                new_y += x_slope * (x_max - new_x);
                new_x = x_max;
            } else if new_x < x_min {
                new_y += x_slope * (x_min - new_x);
                new_x = x_min;
            }
            return pt!(new_x, new_y);
        };

        let a = clip(line.0, line.1);
        let b = clip(line.1, a);

        let ax = a.x;
        let bx = b.x;
        let ay = a.y;
        let by = b.y;

        let dx = bx - ax;
        let dy = by - ay;

        let mut err: f32 = 0.;

        // Case 1: not steep (dx > dy)
        if dx.abs() > dy.abs() {
            let (x0, x1, y0, y1) = if ax < bx {
                (ax, bx, ay, by)
            } else {
                (bx, ax, by, ay)
            };

            let dx = x1 - x0;
            let dy = y1 - y0;
            let derr = (dy / dx).abs();

            let mut y = y0 as u16;

            for x in (x0 as u16)..(x1 as u16) {
                self.plot(Vec2::new(x.into(), y.into()), col);
                err += derr;
                if err > 0.5 {
                    if dy < 0.0 {
                        y -= 1;
                    } else {
                        y += 1;
                    }
                    err -= 1.;
                }
            }
        }
        // Case 2: steep (dx < dy)
        else {
            let (x0, x1, y0, y1) = if ay < by {
                (ax, bx, ay, by)
            } else {
                (bx, ax, by, ay)
            };

            let dx = x1 - x0;
            let dy = y1 - y0;
            let derr = (dx / dy).abs();

            let mut x = x0 as u16;

            for y in (y0 as u16)..(y1 as u16) {
                self.plot(Vec2::new(x.into(), y.into()), col);
                err += derr;
                if err > 0.5 {
                    if dx < 0.0 {
                        x -= 1;
                    } else {
                        x += 1;
                    }
                    err -= 1.;
                }
            }
        }
    }

    fn line3(&mut self, line: &Line3, col: Color) {
        let a = self.pixel(&self.vertex(&line.0));
        let b = self.pixel(&self.vertex(&line.1));

        self.line(&Line2(convpt!(a => Vec2), convpt!(b => Vec2)), col);
    }

    // Placeholder for vertex shader. Eventually should operate on a vertex which has more attributes (color, tex coords) than just a vec3
    // Output coordinate space is camera space
    fn vertex(&self, vert: &Vec3) -> Vec3 {
        let v4 = na::Vector4::new(vert.x, vert.y, vert.z, 1.0);
        let p_camera = convpt!(self.camera.transform_matrix() * v4 => Vec3);
        pt!(
            self.camera.near_clip_plane * p_camera.x / -p_camera.z,
            self.camera.near_clip_plane * p_camera.y / -p_camera.z,
            -p_camera.z
        )
    }

    // Convert camera space to pixel space
    fn pixel(&self, p: &Vec3) -> Vec3 {
        let aspect = self.width as f32 / self.height as f32;
        let canvas_width = (self.camera.hfov / 2.0).tan() * self.camera.near_clip_plane;
        let canvas_height = canvas_width / aspect;

        let normalized = pt!(
            (p.x + canvas_width / 2.0) / canvas_width,
            (p.y + canvas_height / 2.0) / canvas_height,
            p.z
        );

        pt!(
            normalized.x * self.width as f32,
            (1.0 - normalized.y) * self.height as f32,
            p.z
        )
    }

    fn wireframe(&mut self, mesh: &Vec<Tri>, col: Color) {
        for tri in mesh.iter() {
            self.line3(&Line3(tri.0, tri.1), col);
            self.line3(&Line3(tri.1, tri.2), col);
            self.line3(&Line3(tri.2, tri.0), col);
        }
    }
}
