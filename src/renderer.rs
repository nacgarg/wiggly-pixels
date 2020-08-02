extern crate nalgebra as na;
extern crate tobj;

type Color = na::Vector3<u8>;
type Vec2 = na::Vector2<f32>;
type Vec3 = na::Vector3<f32>;
#[derive(Clone, Copy)]
pub struct Tri(Vec3, Vec3, Vec3);
pub type Mesh = Vec<Tri>;

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

pub struct FragParams<'a> {
    // location of pixel in screen/camera space
    p: Vec3,

    // barycentric coords
    u: f32,
    v: f32,
    w: f32,

    // triangle in screen/camera space
    tri: &'a Tri,

    // scene time in seconds
    t: f32,
}

pub struct Camera {
    pub pos: Vec3,
    pub rot: na::Rotation3<f32>,
    pub near_clip_plane: f32,
    pub hfov: f32,
}

pub struct Object {
    pub pos: Vec3,
    pub rot: na::Rotation3<f32>,
    pub mesh: Mesh,
    pub shader: Box<dyn Fn(&mut Renderer, FragParams) -> Option<Color>>,
}

pub struct Renderer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    z_buffer: Vec<f32>,
    state: RenderState,
}

pub struct Scene {
    pub camera: Camera,
    pub objects: Vec<Object>,
}

pub fn rotate_mesh(mesh: &Mesh, rot: na::Rotation3<f32>) -> Mesh {
    let mut ret = mesh.clone();
    for tri in ret.iter_mut() {
        tri.0 = rot * tri.0;
        tri.1 = rot * tri.1;
        tri.2 = rot * tri.2;
    }
    ret.to_vec()
}

impl Object {
    pub fn new<F: 'static>(mesh: Mesh, frag: F) -> Object
    where
        F: Fn(&mut Renderer, FragParams) -> Option<Color>,
    {
        Object {
            pos: Vec3::new(0.0, 0.0, 0.0),
            rot: na::Rotation3::identity(),
            mesh,
            shader: Box::new(frag),
        }
    }
}

impl Camera {
    // Returns the world to camera transformation matrix
    fn transform_matrix(&self) -> na::Matrix4<f32> {
        let translation = na::Translation3::from(self.pos).inverse();
        translation.to_homogeneous() * self.rot.to_homogeneous()
    }
}

impl Tri {
    fn y_sorted(&self) -> Tri {
        let mut tmp = vec![self.0, self.1, self.2];
        tmp.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
        assert!(tmp[0].y <= tmp[1].y);
        assert!(tmp[1].y <= tmp[2].y);
        Tri(tmp[0], tmp[1], tmp[2])
    }
    fn rounded(&self) -> Tri {
        let round = |v: Vec3| -> Vec3 { pt!(v.x.round(), v.y.round(), v.z) };
        Tri(round(self.0), round(self.1), round(self.2))
    }
}

impl Scene {
    pub fn new() -> Self {
        let mut ret = Self {
            camera: Camera {
                pos: pt!(0.0, 0.0, 50.0), // In front of XY plane
                rot: na::Rotation3::from_axis_angle(&Vec3::x_axis(), std::f32::consts::FRAC_PI_2), // facing negative z
                near_clip_plane: 0.1,
                hfov: 1.0,
            },
            objects: Vec::new(),
        };
        ret.init();
        ret
    }
    pub fn init(&mut self) {
        let a = pt!(0.0, 0.0, 0.0);
        let b = pt!(1.0, 0.0, 0.0);
        let c = pt!(0.5, 0.0, 1.0);
        let d = pt!(0.5, 1.0, 0.5);

        let tetra: Mesh = vec![Tri(a, b, c), Tri(a, b, d), Tri(a, d, c), Tri(d, b, c)];
        self.objects
            .push(Object::new(tetra, |s, f| -> Option<Color> {
                // Simple fragment shader that does a z test, writes to z buffer, and does some simple depth shading
                let col = Color::new(20, 140, 180);
                let index = f.p.y as usize * s.width + f.p.x as usize;
                if s.z_buffer[index] <= f.p.z {
                    None
                } else {
                    s.z_buffer[index] = f.p.z;
                    Some(Color::new(
                        (col.x as f32 * f.p.z / 10.0) as u8,
                        (col.y as f32 * f.p.z / 10.0) as u8,
                        (col.z as f32 * f.p.z / 10.0) as u8,
                    ))
                }
            }));

        let (models, _) = tobj::load_obj("cow.obj", false).expect("Failed to load file");
        let m = &models[0].mesh;
        let mut mesh = Mesh::new();
        let mut next_face = 0;
        for f in 0..m.num_face_indices.len() {
            let end = next_face + m.num_face_indices[f] as usize;
            let face_indices: Vec<_> = m.indices[next_face..end].iter().collect();
            let vert = |i: usize, j: usize| m.positions[3 * *face_indices[i] as usize + j];

            mesh.push(Tri(
                pt!(vert(0, 0), vert(0, 1), vert(0, 2)),
                pt!(vert(1, 0), vert(1, 1), vert(1, 2)),
                pt!(vert(2, 0), vert(2, 1), vert(2, 2)),
            ));
            next_face = end;
        }

        self.objects
            .push(Object::new(mesh, |s, f| -> Option<Color> {
                let col = Color::new(255, 105, 180);
                let index = f.p.y as usize * s.width + f.p.x as usize;
                if s.z_buffer[index] <= f.p.z {
                    None
                } else {
                    s.z_buffer[index] = f.p.z;
                    let mut normal = (f.tri.1 - f.tri.0).cross(&(f.tri.2 - f.tri.0));
                    normal = normal / normal.norm();
                    let mut light = Vec3::new(f.t.sin(), f.t.cos(), (f.t*0.1).cos());
                    light = light / light.norm();
                    let dot = (normal.dot(&light)).abs();
                    Some(Color::new(
                        (col.x as f32 * (1.0 - dot)) as u8,
                        (col.y as f32 * (1.0 - dot)) as u8,
                        (col.z as f32 * (1.0 - dot)) as u8,
                    ))
                }
            }));
    }
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![0; width * height],
            z_buffer: vec![std::f32::INFINITY; width * height],
            state: RenderState {
                t: 0.0,
                n: 0,
                last_frame_start: std::time::Instant::now(),
            },
        }
    }
    pub fn draw(&mut self, scene: &Scene) -> Result<(), String> {
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

        // Clear display buffer
        for i in self.buffer.iter_mut() {
            *i = 0x00000000; // __RRGGBB
        }
        // Clear z buffer
        for i in self.z_buffer.iter_mut() {
            *i = std::f32::INFINITY;
        }

        let rot = na::Rotation3::from_axis_angle(&Vec3::x_axis(), self.state.t)
            * na::Rotation3::from_axis_angle(&Vec3::y_axis(), self.state.t / 2.0)
            * na::Rotation3::from_axis_angle(&Vec3::z_axis(), self.state.t / 1.5);

        for obj in scene.objects.iter() {
            self.solid(&rotate_mesh(&obj.mesh, rot), &obj.shader, &scene.camera);
            // self.wireframe(&rotate_mesh(&obj.mesh, rot), Color::new(255, 255, 255), &scene.camera);
        }

        // To display z buffer
        // let mut i=0;
        // for a in self.buffer.iter_mut() {
        //     *a = self.z_buffer[i] as u32;
        //     i += 1;
        // }

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

    fn line3(&mut self, line: &Line3, col: Color, camera: &Camera) {
        let a = self.pixel(&self.vertex(&line.0, camera), camera);
        let b = self.pixel(&self.vertex(&line.1, camera), camera);

        self.line(&Line2(convpt!(a => Vec2), convpt!(b => Vec2)), col);
    }

    // Placeholder for vertex shader. Eventually should return a vertex which has more attributes (color, tex coords) than just a vec3
    // Output coordinate space is camera space
    fn vertex(&self, vert: &Vec3, camera: &Camera) -> Vec3 {
        let v4 = na::Vector4::new(vert.x, vert.y, vert.z, 1.0);
        let p_camera = convpt!(camera.transform_matrix() * v4 => Vec3);
        pt!(
            camera.near_clip_plane * p_camera.x / -p_camera.z,
            camera.near_clip_plane * p_camera.y / -p_camera.z,
            -p_camera.z
        )
    }

    // Convert camera space to pixel space
    fn pixel(&self, p: &Vec3, camera: &Camera) -> Vec3 {
        let aspect = self.width as f32 / self.height as f32;
        let canvas_width = (camera.hfov / 2.0).tan() * camera.near_clip_plane;
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

    fn wireframe(&mut self, mesh: &Mesh, col: Color, camera: &Camera) {
        for tri in mesh.iter() {
            self.line3(&Line3(tri.0, tri.1), col, camera);
            self.line3(&Line3(tri.1, tri.2), col, camera);
            self.line3(&Line3(tri.2, tri.0), col, camera);
        }
    }
    // Helper function for rastering a triangle
    fn scanline<F>(&mut self, mut x1: f32, mut x2: f32, y: f32, frag: &F, tri: &Tri)
    where
        F: Fn(&mut Renderer, FragParams) -> Option<Color>,
    {
        if x1 > x2 {
            std::mem::swap(&mut x1, &mut x2);
        }
        if y < 0.0 || y > self.height as f32 - 1.0 {
            return;
        }
        if x1 < 0.0 {
            x1 = 0.0;
        } else if x1 > self.width as f32 - 1.0 {
            x1 = self.width as f32 - 1.0;
        }
        if x2 < 0.0 {
            x2 = 0.0;
        } else if x2 > self.width as f32 - 1.0 {
            x2 = self.width as f32 - 1.0;
        }
        for x in (x1 as u16)..(x2 as u16) {
            // Get barycentric coords of triangle to pass to fragment shader
            // TODO: move this elsewhere so it only happens if we need it in the shader
            // adapted from https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
            let p = pt!(x as f32, y, 0.0);
            let v0 = tri.1 - tri.0;
            let v1 = tri.2 - tri.0;
            let v2 = p - tri.0;

            let d00 = v0.dot(&v0);
            let d01 = v0.dot(&v1);
            let d11 = v1.dot(&v1);
            let d20 = v2.dot(&v0);
            let d21 = v2.dot(&v1);
            let denom = d00 * d11 - (d01 * d01);
            let v = (d11 * d20 - (d01 * d21)) / denom;
            let w = (d00 * d21 - (d01 * d20)) / denom;
            let u = 1.0 - v - w;
            let z = u * tri.0.z + v * tri.1.z + w * tri.2.z;
            let pos = pt!(x as f32, y, z);
            let col = frag(
                self,
                FragParams {
                    p: pos,
                    u,
                    v,
                    w,
                    tri,
                    t: self.state.t,
                },
            );
            match col {
                Some(c) => {
                    self.plot(Vec2::new(x as f32, y), c);
                }
                None => {
                    // don't render anything
                }
            }
        }
    }

    // Raster a triangle using a given fragment shader
    fn tri<F>(&mut self, tri: &Tri, frag: &F, camera: &Camera)
    where
        F: Fn(&mut Renderer, FragParams) -> Option<Color>,
    {
        let tri_proj = Tri(
            self.pixel(&self.vertex(&tri.0, camera), camera),
            self.pixel(&self.vertex(&tri.1, camera), camera),
            self.pixel(&self.vertex(&tri.2, camera), camera),
        );

        let sorted = tri_proj.y_sorted().rounded();

        let bottom_tri = |t: &Tri, s: &mut Renderer| {
            let slope = (
                (t.1.x - t.0.x) / (t.1.y - t.0.y),
                (t.2.x - t.0.x) / (t.2.y - t.0.y),
            );
            let mut x1 = t.0.x;
            let mut x2 = t.0.x;
            let mut y: i16 = t.0.y.round() as i16;
            loop {
                if y >= t.1.y.round() as i16 {
                    break;
                }
                s.scanline(x1.round(), x2.round(), y.into(), frag, &sorted);
                x1 += slope.0;
                x2 += slope.1;
                y += 1;
            }
        };

        let top_tri = |t: &Tri, s: &mut Renderer| {
            let slope = (
                (t.2.x - t.0.x) / (t.2.y - t.0.y),
                (t.2.x - t.1.x) / (t.2.y - t.1.y),
            );
            let mut x1 = t.2.x;
            let mut x2 = t.2.x;
            let mut y: i16 = t.2.y.round() as i16;
            loop {
                if y < t.0.y.round() as i16 {
                    break;
                }
                s.scanline(x1.round(), x2.round(), y.into(), frag, &sorted);
                x1 -= slope.0;
                x2 -= slope.1;
                y -= 1;
            }
        };

        if sorted.1.y.round() == sorted.2.y.round() {
            bottom_tri(&sorted, self);
        } else if sorted.0.y.round() == sorted.1.y.round() {
            top_tri(&sorted, self);
        } else {
            let intersection = if sorted.2.x < sorted.0.x {
                (sorted.0.x
                    + (sorted.1.y - sorted.0.y) / (sorted.2.y - sorted.0.y)
                        * (sorted.2.x - sorted.0.x))
                    .round()
            } else {
                (sorted.0.x
                    + (sorted.1.y - sorted.0.y) / (sorted.2.y - sorted.0.y)
                        * (sorted.2.x - sorted.0.x))
                    .round()
            };
            let split = pt!(intersection, sorted.1.y.round(), 0.0);
            bottom_tri(&Tri(sorted.0, sorted.1, split), self);
            top_tri(&Tri(split, sorted.1, sorted.2), self);
        }
    }
    fn solid<F>(&mut self, mesh: &Mesh, frag: &F, camera: &Camera)
    where
        F: Fn(&mut Renderer, FragParams) -> Option<Color>,
    {
        for tri in mesh.iter() {
            self.tri(tri, frag, camera);
        }
    }
}
