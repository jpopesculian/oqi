pub struct Newlines {
    /// Byte offsets of every `\n`.
    pub all: Vec<usize>,
    /// Byte offsets of `\n` characters that terminate blank (whitespace-only) lines.
    pub blank: Vec<usize>,
}

pub fn find_newlines(source: &str) -> Newlines {
    let mut all = Vec::new();
    let mut blank = Vec::new();
    let mut line_is_blank = true;
    for (i, b) in source.bytes().enumerate() {
        if b == b'\n' {
            all.push(i);
            if line_is_blank {
                blank.push(i);
            }
            line_is_blank = true;
        } else if !b.is_ascii_whitespace() {
            line_is_blank = false;
        }
    }
    Newlines { all, blank }
}
