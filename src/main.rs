extern crate minifb;
use minifb::{Key, Window, WindowOptions};

mod renderer;
use renderer::{Renderer, Scene};

static CAP_FPS: bool = true;

fn main() {
    let mut r = Renderer::new(1920, 1080);
    let mut s = Scene::new();
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
        if window.is_key_down(Key::W) {
            s.camera.pos.z -= 0.1;
        }
        if window.is_key_down(Key::A) {
            s.camera.pos.x -= 0.1;
        }
        if window.is_key_down(Key::S) {
            s.camera.pos.z += 0.1;
        }
        if window.is_key_down(Key::D) {
            s.camera.pos.x += 0.1;
        }
        if window.is_key_down(Key::Q) {
            s.camera.hfov += 0.01;
        }
        if window.is_key_down(Key::E) {
            s.camera.hfov -= 0.01;
        }

        r.draw(&s).unwrap_or_else(|err| {
            panic!("Error in renderer: {}", err);
        });
        
        window
            .update_with_buffer(&r.buffer, r.width, r.height)
            .unwrap_or_else(|err| {
                panic!("Error updating window buffer: {}", err);
            });
    }
}
