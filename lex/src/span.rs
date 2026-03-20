use std::ops::{Index, Range};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[inline]
    pub const fn offset(self, offset: usize) -> Self {
        Self {
            start: self.start + offset,
            end: self.end + offset,
        }
    }

    pub fn doc_position(self, doc: &str, tab_size: usize) -> (usize, usize) {
        let mut line = 1;
        let mut column = 1;
        for (i, c) in doc.char_indices() {
            if i >= self.start {
                break;
            }
            match c {
                '\n' => {
                    line += 1;
                    column = 1;
                }
                '\t' => {
                    column += tab_size;
                }
                _ => {
                    column += 1;
                }
            }
        }
        (line, column)
    }
}

pub const fn span(start: usize, end: usize) -> Span {
    Span::new(start, end)
}

impl From<Range<usize>> for Span {
    fn from(range: Range<usize>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

impl From<Span> for Range<usize> {
    fn from(span: Span) -> Self {
        span.start..span.end
    }
}

impl Index<Span> for str {
    type Output = <str as Index<Range<usize>>>::Output;
    fn index(&self, span: Span) -> &Self::Output {
        &self[Range::<usize>::from(span)]
    }
}
