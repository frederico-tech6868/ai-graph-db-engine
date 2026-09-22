//! Rendering for the TUI.

use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use ratatui::Frame;

use crate::app::{App, Mode};

/// Draw the whole UI for the current frame.
pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // tabs
            Constraint::Min(3),    // output
            Constraint::Length(3), // input
            Constraint::Length(1), // footer
        ])
        .split(f.area());

    // Tabs (modes).
    let titles: Vec<Line> = Mode::ALL
        .iter()
        .map(|m| Line::from(Span::raw(m.title())))
        .collect();
    let tabs = Tabs::new(titles)
        .select(app.mode.index())
        .block(Block::default().borders(Borders::ALL).title(" graphdb-tui "))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(tabs, chunks[0]);

    // Output area — title changes per mode to show the active model in Chat.
    let output_title = match app.mode {
        Mode::Chat => format!(" Chat — model: {} ", app.active_model_name()),
        _ => format!(" Output — engine: {} ", app.engine),
    };
    let output = Paragraph::new(app.output.as_str())
        .block(Block::default().borders(Borders::ALL).title(output_title))
        .wrap(Wrap { trim: false });
    f.render_widget(output, chunks[1]);

    // Input area.
    let input = Paragraph::new(app.input.as_str()).block(
        Block::default()
            .borders(Borders::ALL)
            .title(input_title(app.mode)),
    );
    f.render_widget(input, chunks[2]);

    // Cursor position at the end of the input text.
    let cursor_x = chunks[2].x + 1 + app.input.chars().count() as u16;
    let cursor_y = chunks[2].y + 1;
    f.set_cursor_position((cursor_x, cursor_y));

    // Footer help line — Ctrl+M hint only shown when a model toggle is possible.
    let mut footer_spans = vec![
        Span::styled("Tab", Style::default().fg(Color::Yellow)),
        Span::raw(" mode  "),
        Span::styled("Ctrl+E", Style::default().fg(Color::Yellow)),
        Span::raw(" engine  "),
    ];
    if app.can_toggle_model() {
        footer_spans.push(Span::styled("Ctrl+M", Style::default().fg(Color::Yellow)));
        footer_spans.push(Span::raw(" model  "));
    }
    footer_spans.extend([
        Span::styled("Enter", Style::default().fg(Color::Yellow)),
        Span::raw(" submit  "),
        Span::styled("Esc/Ctrl+Q", Style::default().fg(Color::Yellow)),
        Span::raw(" quit"),
    ]);
    let footer = Paragraph::new(Line::from(footer_spans)).dim();
    f.render_widget(footer, chunks[3]);
}

fn input_title(mode: Mode) -> String {
    let hint = match mode {
        Mode::Chat => "message",
        Mode::Rag => "question",
        Mode::Extract => "text to extract from",
        Mode::Info => "press Enter to refresh",
    };
    format!(" Input ({hint}) ")
}
