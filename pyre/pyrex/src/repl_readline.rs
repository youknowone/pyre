use std::io::{self, IsTerminal, Write};
use std::path::Path;

use rustyline::completion::Completer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{CompletionType, Config, Context, Editor, Helper};

pub enum ReadlineResult {
    Line(String),
    Eof,
    Interrupt,
    Io(std::io::Error),
    Other(String),
}

#[derive(Default)]
struct EmptyHelper;

impl Completer for EmptyHelper {
    type Candidate = String;

    fn complete(
        &self,
        _line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        Ok((pos, Vec::new()))
    }
}

impl Hinter for EmptyHelper {
    type Hint = String;
}

impl Highlighter for EmptyHelper {}
impl Validator for EmptyHelper {}
impl Helper for EmptyHelper {}

enum ReadlineBackend {
    Basic,
    Editor(Editor<EmptyHelper, DefaultHistory>),
}

pub struct Readline {
    backend: ReadlineBackend,
}

impl Readline {
    pub fn new() -> Self {
        if io::stdin().is_terminal() && io::stdout().is_terminal() {
            let mut editor = Editor::with_config(
                Config::builder()
                    .completion_type(CompletionType::List)
                    .tab_stop(8)
                    .bracketed_paste(false)
                    .build(),
            )
            .expect("failed to initialize line editor");
            editor.set_helper(Some(EmptyHelper));
            Self {
                backend: ReadlineBackend::Editor(editor),
            }
        } else {
            Self {
                backend: ReadlineBackend::Basic,
            }
        }
    }

    pub fn load_history(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        match &mut self.backend {
            ReadlineBackend::Basic => Ok(()),
            ReadlineBackend::Editor(editor) => {
                editor.load_history(path)?;
                Ok(())
            }
        }
    }

    pub fn save_history(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        match &mut self.backend {
            ReadlineBackend::Basic => Ok(()),
            ReadlineBackend::Editor(editor) => {
                if !path.exists()
                    && let Some(parent) = path.parent()
                {
                    std::fs::create_dir_all(parent)?;
                }
                editor.save_history(path)?;
                Ok(())
            }
        }
    }

    pub fn add_history_entry(&mut self, entry: &str) -> Result<(), Box<dyn std::error::Error>> {
        match &mut self.backend {
            ReadlineBackend::Basic => Ok(()),
            ReadlineBackend::Editor(editor) => {
                editor.add_history_entry(entry)?;
                Ok(())
            }
        }
    }

    pub fn readline(&mut self, prompt: &str) -> ReadlineResult {
        match &mut self.backend {
            ReadlineBackend::Basic => read_basic_line(prompt),
            ReadlineBackend::Editor(editor) => {
                use rustyline::error::ReadlineError;

                match editor.readline(prompt) {
                    Ok(line) => ReadlineResult::Line(line),
                    Err(ReadlineError::Interrupted) => ReadlineResult::Interrupt,
                    Err(ReadlineError::Eof) => ReadlineResult::Eof,
                    Err(ReadlineError::Io(err)) => ReadlineResult::Io(err),
                    Err(err) => ReadlineResult::Other(err.to_string()),
                }
            }
        }
    }
}

fn read_basic_line(prompt: &str) -> ReadlineResult {
    print!("{prompt}");
    if let Err(err) = io::stdout().flush() {
        return ReadlineResult::Io(err);
    }

    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) => ReadlineResult::Eof,
        Ok(_) => {
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            ReadlineResult::Line(line)
        }
        Err(err) if err.kind() == io::ErrorKind::Interrupted => ReadlineResult::Interrupt,
        Err(err) => ReadlineResult::Io(err),
    }
}
