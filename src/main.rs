extern crate minifb;
use minifb::{Key, Window, WindowOptions};

mod renderer;
use renderer::*;

fn main() {
    let r = Renderer {
        width: 1920,
        height: 1080,
    };
    let mut buffer: Vec<u32> = vec![0; r.width * r.height];

    let mut window = Window::new(
        "Wiggly Renderer",
        r.width,
        r.height,
        WindowOptions::default(),
    )
    .unwrap_or_else(|err| {
        panic!("Error opening window: {}", err);
    });

    println!("Starting window event loop with renderer: {:?}", r);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        r.render(&mut buffer).unwrap_or_else(|err| {
            panic!("Error in renderer {:?}: {}", r, err);
        });

        window
            .update_with_buffer(&buffer, r.width, r.height)
            .unwrap_or_else(|err| {
                panic!("Error updating window buffer: {}", err);
            });
    }
}
