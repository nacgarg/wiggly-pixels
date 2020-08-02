extern crate minifb;
use minifb::{Key, Window, WindowOptions};

mod renderer;
use renderer::Renderer;

static CAP_FPS: bool = true;

fn main() {
    let mut r = Renderer::new(1920, 1080);
    let mut window = Window::new(
        "Wiggly Renderer",
        r.width,
        r.height,
        WindowOptions {
            scale_mode: minifb::ScaleMode::UpperLeft,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|err| {
        panic!("Error opening window: {}", err);
    });

    println!("Starting window event loop");

    if CAP_FPS {
        println!("Limitng FPS to 60");
        window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));
    }

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
