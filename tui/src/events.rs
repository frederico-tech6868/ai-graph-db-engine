//! Key-event handling for the TUI.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::App;

/// What the event loop should do after handling a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Nothing further; just re-render.
    None,
    /// The user submitted the input line (run the async handler).
    Submit,
    /// Quit the application.
    Quit,
}

/// Handle a single key event, mutating `app` and returning the next [`Action`].
pub fn handle_key(app: &mut App, key: KeyEvent) -> Action {
    // Global shortcuts.
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('c') => return Action::Quit,
            KeyCode::Char('e') => {
                app.toggle_engine();
                return Action::None;
            }
            _ => {}
        }
    }

    match key.code {
        KeyCode::Esc => Action::Quit,
        KeyCode::Tab => {
            app.next_mode();
            Action::None
        }
        KeyCode::Enter => Action::Submit,
        KeyCode::Backspace => {
            app.input.pop();
            Action::None
        }
        KeyCode::Char(c) => {
            app.input.push(c);
            Action::None
        }
        _ => Action::None,
    }
}
