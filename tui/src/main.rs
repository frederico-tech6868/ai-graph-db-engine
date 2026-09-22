//! `graphdb-tui` — a terminal UI for the model-hub AI pipelines.
//!
//! Offers four modes (Chat, RAG, Extract, System) over a demo knowledge base,
//! running fully offline with stub models. The tool-calling / extraction /
//! embedding engine can be toggled at runtime between LLM and Needle (Ctrl+E).

mod app;
mod events;
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

#[tokio::main]
async fn main() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new();

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
