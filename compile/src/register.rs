//! Allocation of the flat global quantum memory and the logical registers
//! over it: a register is an ordered list of ranges of global qubit
//! indices, which slicing, index sets and `++` concatenation rearrange.
//! Only the qubit layout ([`crate::qubits`]) uses these.

use std::ops::Range;

/// Allocator for the single flat global quantum memory. OpenQASM qubits
/// are all statically allocated, in declaration order, and never freed or
/// relocated, so allocation is just a bump of a running cursor: each
/// [`alloc`](Self::alloc) hands out the next contiguous span of global
/// indices.
#[derive(Debug, Clone, Default)]
pub(crate) struct QuantumMemory {
    size: usize,
}

impl QuantumMemory {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Total number of qubits allocated so far.
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Allocate `size` fresh qubits and return them as a contiguous
    /// register over global indices.
    #[allow(clippy::single_range_in_vec_init)]
    pub(crate) fn alloc(&mut self, size: usize) -> QuantumRegister {
        let start = self.size;
        self.size += size;
        QuantumRegister {
            ranges: vec![start..start + size],
        }
    }
}

/// A logical quantum register: an ordered list of half-open ranges of
/// **global** qubit indices. A plain declaration is one contiguous range;
/// slicing, discrete selection, and `++` concatenation build registers
/// with several ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct QuantumRegister {
    ranges: Vec<Range<usize>>,
}

impl QuantumRegister {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The global index ranges making up this register, in logical order.
    pub(crate) fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    pub(crate) fn len(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).sum()
    }

    /// The register covering logical positions `range` of this one.
    pub(crate) fn slice(&self, range: Range<usize>) -> QuantumRegister {
        assert!(range.start <= range.end, "slice start exceeds end");
        assert!(range.end <= self.len(), "slice end exceeds register length");

        let mut ranges = Vec::new();
        let mut skip = range.start;
        let mut remaining = range.end - range.start;

        for part in &self.ranges {
            if remaining == 0 {
                break;
            }
            let part_len = part.len();
            if skip >= part_len {
                skip -= part_len;
                continue;
            }
            let take = remaining.min(part_len - skip);
            let start = part.start + skip;
            ranges.push(start..(start + take));
            remaining -= take;
            skip = 0;
        }

        QuantumRegister { ranges }
    }

    pub(crate) fn concat(mut self, other: Self) -> Self {
        self.ranges.extend(other.ranges);
        self
    }

    /// Global index of the qubit at logical position `idx`.
    pub(crate) fn global_index_of(&self, mut idx: usize) -> Option<usize> {
        for range in &self.ranges {
            let n = range.len();
            if idx < n {
                return Some(range.start + idx);
            }
            idx -= n;
        }
        None
    }

    /// Iterate the register's global qubit indices in logical order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().cloned().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn alloc_returns_register_with_single_range() {
        let mut mem = QuantumMemory::new();
        let reg = mem.alloc(3);
        assert_eq!(reg.ranges(), &[0..3]);
        assert_eq!(reg.len(), 3);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn allocations_advance_the_global_cursor() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(2);
        assert_eq!(a.ranges(), &[0..3]);
        assert_eq!(b.ranges(), &[3..5]);
        assert_eq!(mem.size(), 5);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn registers_carry_global_ranges() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(2);
        let c = mem.alloc(4);

        assert_eq!(a.ranges(), &[0..3]);
        assert_eq!(b.ranges(), &[3..5]);
        assert_eq!(c.ranges(), &[5..9]);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn slice_within_single_range() {
        let mut mem = QuantumMemory::new();
        let reg = mem.alloc(5);
        let s = reg.slice(1..4);
        assert_eq!(s.ranges(), &[1..4]);
    }

    #[test]
    fn slice_spans_multiple_ranges() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let joined = a.concat(b);
        let s = joined.slice(2..6);
        assert_eq!(s.ranges(), &[2..3, 3..6]);
    }

    #[test]
    fn concat_appends_ranges() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(2);
        let b = mem.alloc(3);
        let joined = a.concat(b);
        assert_eq!(joined.ranges(), &[0..2, 2..5]);
        assert_eq!(joined.len(), 5);
    }

    #[test]
    fn slice_of_concat_yields_expected_global_indices() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let joined = a.concat(b);
        let indices: Vec<usize> = joined.slice(2..6).iter().collect();
        assert_eq!(indices, vec![2, 3, 4, 5]);
    }

    #[test]
    fn register_iter_yields_global_indices() {
        let mut mem = QuantumMemory::new();
        let _ = mem.alloc(3);
        let b = mem.alloc(2);
        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![3, 4]);
    }

    #[test]
    fn global_index_of_returns_global_index() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let reg = a.concat(b);
        assert_eq!(reg.global_index_of(0), Some(0));
        assert_eq!(reg.global_index_of(3), Some(3));
        assert_eq!(reg.global_index_of(6), Some(6));
        assert_eq!(reg.global_index_of(7), None);
    }
}
