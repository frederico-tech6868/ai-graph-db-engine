//! `graphdb-tui` — a terminal UI for the model-hub AI pipelines.
//!
//! Offers four modes (Chat, RAG, Extract, System) over a demo knowledge base.
//! On startup, reads `graphdb-agent.settings.json` (same discovery order as
//! `graphdb-cli agent`) and, if a GGUF model path is configured, loads it as
//! the primary text model. The offline stub is always available as a fallback
//! and can be toggled at runtime with Ctrl+M (Chat mode only).

mod app;
mod events;
mod settings;
mod ui;

use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use model_hub::models::text::StubTextModel;
use model_hub::GenerationConfig;

use crate::app::App;
use crate::events::{handle_key, Action};

type Tui = Terminal<CrosstermBackend<Stdout>>;

fn setup_terminal() -> Result<Tui> {
    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(terminal: &mut Tui) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

/// Try to load the GGUF model described in `settings`.
///
/// Returns `Some(model)` on success, `None` if the path is not configured or
/// loading fails (a warning is printed to stderr before raw mode is entered).
fn try_load_model(
    s: &settings::TuiSettings,
) -> Option<Box<dyn model_hub::TextModel>> {
    let gguf = s.model_path()?;
    let tok = s.tokenizer_path()?;

    eprintln!("graphdb-tui: loading model {} …", gguf.display());

    let device = match model_hub::resolve_device(s.device_kind()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("warning: device resolution failed ({e}); falling back to stub");
            return None;
        }
    };

    match model_hub::models::text::QuantizedLlama::load(&gguf, &tok, device) {
        Ok(m) => {
            eprintln!("graphdb-tui: model loaded successfully.");
            Some(Box::new(m))
        }
        Err(e) => {
            eprintln!(
                "warning: failed to load model {} — {e}; falling back to stub",
                gguf.display()
            );
            None
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── 1. Load settings and model BEFORE entering raw-mode so warnings are
    //       visible in the normal terminal output. ──────────────────────────
    let (s, source) = settings::TuiSettings::load();

    let gen = if s.generation.temperature <= 0.0 {
        GenerationConfig::deterministic().with_max_tokens(s.generation.max_tokens)
    } else {
        GenerationConfig::default()
            .with_max_tokens(s.generation.max_tokens)
            .with_temperature(s.generation.temperature)
    };

    // If the settings point to a real GGUF model, that becomes primary and the
    // stub becomes the Ctrl+M alternate.  Otherwise the stub is primary and
    // Ctrl+M is hidden.
    let (primary, alt): (Box<dyn model_hub::TextModel>, Option<Box<dyn model_hub::TextModel>>) =
        match try_load_model(&s) {
            Some(real) => (real, Some(Box::new(StubTextModel::new()))),
            None => (Box::new(StubTextModel::new()), None),
        };

    // ── 2. Enter raw / alternate-screen mode. ────────────────────────────────
    let mut terminal = setup_terminal()?;
    let mut app = App::with_models(primary, alt, gen, source);

    let res = run(&mut terminal, &mut app).await;

    restore_terminal(&mut terminal)?;
    res
}

async fn run(terminal: &mut Tui, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        // Poll for input with a small timeout so the loop stays responsive.
        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == event::KeyEventKind::Press {
                    match handle_key(app, key) {
                        Action::Quit => {
                            app.should_quit = true;
                        }
                        Action::Submit => {
                            app.submit().await;
                        }
                        Action::None => {}
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }
    Ok(())
}
