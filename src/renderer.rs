extern crate nalgebra as na;
#[derive(Debug)]
pub struct Renderer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    t: u64,
}

type Color = na::Vector3<u8>;
type Point2 = na::Vector2<f32>;

#[derive(Debug)]
struct Line2 {
    a: Point2,
    b: Point2,
    color: Color,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Renderer {
        Renderer {
            width,
            height,
            buffer: vec![0; width * height],
            t: 0,
        }
    }
    pub fn draw(&mut self) -> Result<(), String> {
        let start = std::time::Instant::now();

        for i in self.buffer.iter_mut() {
            // Draw blue
            *i = 0x000000FF; // __RRGGBB
        }
        // for i in 0..self.width {
        //     for j in 0..self.height {
        //         self.plot(Point2::new(i as f32, j as f32), Color::new(255, 255, 0));
        //     }
        // }

        let t = self.t;
        self.line(&Line2 {
            a: Point2::new(300.0, 300.0),
            b: Point2::new(
                300.0 + 300.0 * (t as f32 / 100.0).cos(),
                300.0 + 300.0 * (t as f32 / 100.0).sin(),
            ),
            color: Color::new(255, 255, 0),
        });
        self.t += 1;
        if t % 100 == 0 {
            println!(
                "draw: {:?} ({} fps)",
                start.elapsed(),
                1.0 / (start.elapsed().as_secs_f32())
            );
        }
        Ok(())
    }

    // Plots a single pixel with the given color
    fn plot(&mut self, pos: Point2, col: Color) {
        let index = pos.y as usize * self.width + pos.x as usize;
        let pixel = &mut self.buffer[index];
        *pixel = (col.x as u32) << 16 as u32;
        *pixel |= (col.x as u32) << 8 as u32;
        *pixel |= col.z as u32;
    }

    fn line(&mut self, line: &Line2) {
        let ax = line.a.x;
        let bx = line.b.x;
        let ay = line.a.y;
        let by = line.b.y;

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
                self.plot(Point2::new(x.into(), y.into()), line.color);
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
                self.plot(Point2::new(x.into(), y.into()), line.color);
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
}
