#[derive(Debug)]
pub struct Renderer {
    pub width: usize,
    pub height: usize,
}

impl Renderer {
    pub fn render(&self, buffer: &mut Vec<u32>) -> Result<(), String> {
        for i in buffer.iter_mut() {
            // Draw blue
            *i = 0x000000FF; // __RRGGBB
        }
        Ok(())
    }
}
