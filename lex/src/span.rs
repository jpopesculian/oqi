use std::ops::{Index, Range};

/// A half-open range `start..end` of byte offsets into a source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Span {
    /// Offset of the first byte.
    pub start: usize,
    /// Offset one past the last byte.
    pub end: usize,
}

impl Span {
    /// The span `start..end`.
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// The span from the start of `self` to the end of `end`.
    #[inline]
    pub const fn join(self, end: Span) -> Self {
        Self {
            start: self.start,
            end: end.end,
        }
    }

    /// The same span, `offset` bytes later.
    #[inline]
    pub const fn offset(self, offset: usize) -> Self {
        Self {
            start: self.start + offset,
            end: self.end + offset,
        }
    }

    /// The number of bytes the span covers.
    #[inline]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the span covers no bytes.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.end <= self.start
    }

    /// Whether the byte at `offset` lies in the span.
    #[inline]
    pub const fn contains(self, offset: usize) -> bool {
        self.start <= offset && offset < self.end
    }
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
