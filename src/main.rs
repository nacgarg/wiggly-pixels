extern crate minifb;
use minifb::{Key, Window, WindowOptions};

mod renderer;
use renderer::Renderer;

fn main() {
    let mut r = Renderer::new(1920, 1080);
    let mut window = Window::new(
        "Wiggly Renderer",
        r.width,
        r.height,
        WindowOptions::default(),
    )
    .unwrap_or_else(|err| {
        panic!("Error opening window: {}", err);
    });

    println!("Starting window event loop");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        r.draw().unwrap_or_else(|err| {
            panic!("Error in renderer {:?}: {}", r, err);
        });

        window
            .update_with_buffer(&r.buffer, r.width, r.height)
            .unwrap_or_else(|err| {
                panic!("Error updating window buffer: {}", err);
            });
    }
}
