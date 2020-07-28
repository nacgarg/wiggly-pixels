extern crate nalgebra as na;
#[derive(Debug)]
pub struct Renderer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
}

type Color = na::Vector3<u8>;
type Point2 = na::Vector2<f32>;

impl Renderer {
    pub fn new(width: usize, height: usize) -> Renderer {
        Renderer {
            width,
            height,
            buffer: vec![0; width * height],
        }
    }
    pub fn draw(&mut self) -> Result<(), String> {
        // for i in self.buffer.iter_mut() {
        //     // Draw blue
        //     *i = 0x000000FF; // __RRGGBB
        // }
        for i in 0..self.width {
            for j in 0..self.height {
                self.plot(Point2::new(i as f32, j as f32), Color::new(255, 255, 0));
            }
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
}
