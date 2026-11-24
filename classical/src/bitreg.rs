use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, Not, Range, Shl, Shr};

/// The bits of a `bit[n]` register, of any width. Bit `i` is the `i`-th
/// least significant; positions past the stored bits read as `0`. The
/// register does not know its width — that lives in the type
/// ([`PrimitiveTy::BitReg`](crate::PrimitiveTy::BitReg)).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitReg(Repr);

/// Up to 128 bits inline, more on the heap (little-endian bytes).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Repr {
    Stack(u128),
    Heap(Box<[u8]>),
}

const STACK_BYTES: usize = 16;

#[inline]
fn ceil_div_8(width: u32) -> usize {
    (width as usize).div_ceil(8)
}

impl BitReg {
    const fn stack(bits: u128) -> Self {
        Self(Repr::Stack(bits))
    }

    fn heap(bytes: Box<[u8]>) -> Self {
        Self(Repr::Heap(bytes))
    }

    /// A register of `width` zero bits.
    pub fn zeros(width: u32) -> Self {
        if width <= 128 {
            BitReg::stack(0)
        } else {
            BitReg::heap(vec![0u8; ceil_div_8(width)].into_boxed_slice())
        }
    }

    /// Bit `pos`; `false` past the stored bits.
    pub fn get_bit(&self, pos: usize) -> bool {
        match &self.0 {
            Repr::Stack(v) => {
                if pos >= 128 {
                    false
                } else {
                    (v >> pos) & 1 != 0
                }
            }
            Repr::Heap(bytes) => {
                let byte_idx = pos / 8;
                if byte_idx >= bytes.len() {
                    false
                } else {
                    (bytes[byte_idx] >> (pos % 8)) & 1 != 0
                }
            }
        }
    }

    /// Set bit `pos`. Positions past the stored bits are ignored: size the
    /// register with [`zeros`](Self::zeros) or [`resize`](Self::resize).
    pub fn set_bit(&mut self, pos: usize, value: bool) {
        match &mut self.0 {
            Repr::Stack(v) => {
                if pos >= 128 {
                    return;
                }
                let mask = 1u128 << pos;
                if value { *v |= mask } else { *v &= !mask }
            }
            Repr::Heap(bytes) => {
                let byte_idx = pos / 8;
                if byte_idx >= bytes.len() {
                    return;
                }
                let mask = 1u8 << (pos % 8);
                if value {
                    bytes[byte_idx] |= mask
                } else {
                    bytes[byte_idx] &= !mask
                }
            }
        }
    }

    pub(crate) fn iter_bits(&self) -> BitIter<'_> {
        BitIter { reg: self, pos: 0 }
    }

    pub(crate) fn get_bits(&self, positions: impl IntoIterator<Item = usize>) -> Self {
        let positions: Vec<usize> = positions.into_iter().collect();
        let mut out = Self::zeros(positions.len() as u32);
        for (i, pos) in positions.into_iter().enumerate() {
            out.set_bit(i, self.get_bit(pos));
        }
        out
    }

    /// Set each `(position, value)` in turn, as [`set_bit`](Self::set_bit).
    pub fn set_bits(&mut self, ops: impl IntoIterator<Item = (usize, bool)>) {
        for (pos, value) in ops {
            self.set_bit(pos, value);
        }
    }

    /// Rotate the bits at `slice` left (towards higher positions) by
    /// `amount`. An empty or reversed slice is left alone.
    pub(crate) fn rotl(&mut self, slice: Range<u32>, amount: usize) {
        let len = slice.end.saturating_sub(slice.start) as usize;
        if len == 0 {
            return;
        }
        let amount = amount % len;
        if amount == 0 {
            return;
        }
        let bits: Vec<bool> = (slice.start..slice.end)
            .map(|p| self.get_bit(p as usize))
            .collect();
        for (i, &bit) in bits.iter().enumerate() {
            let new_pos = slice.start as usize + (i + amount) % len;
            self.set_bit(new_pos, bit);
        }
    }

    /// The low 128 bits as an integer.
    pub fn as_u128(&self) -> u128 {
        match &self.0 {
            Repr::Stack(v) => *v,
            Repr::Heap(bytes) => {
                let mut buf = [0u8; STACK_BYTES];
                let n = bytes.len().min(STACK_BYTES);
                buf[..n].copy_from_slice(&bytes[..n]);
                u128::from_le_bytes(buf)
            }
        }
    }

    /// The register cut or zero-extended to `new_width` bits.
    pub fn resize(self, new_width: u32) -> Self {
        if new_width <= 128 {
            let v = self.as_u128();
            let masked = if new_width == 128 {
                v
            } else if new_width == 0 {
                0
            } else {
                v & ((1u128 << new_width) - 1)
            };
            BitReg::stack(masked)
        } else {
            let n_bytes = ceil_div_8(new_width);
            let mut buf = vec![0u8; n_bytes];
            match &self.0 {
                Repr::Stack(v) => {
                    let src = v.to_le_bytes();
                    let copy_len = STACK_BYTES.min(n_bytes);
                    buf[..copy_len].copy_from_slice(&src[..copy_len]);
                }
                Repr::Heap(bytes) => {
                    let copy_len = bytes.len().min(n_bytes);
                    buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
                }
            }
            let trailing = new_width % 8;
            if trailing != 0 {
                let last = n_bytes - 1;
                buf[last] &= (1u8 << trailing) - 1;
            }
            BitReg::heap(buf.into_boxed_slice())
        }
    }

    /// The number of bits stored: 128 inline, or the heap allocation's.
    pub(crate) fn default_width(&self) -> u32 {
        match &self.0 {
            Repr::Stack(_) => 128,
            Repr::Heap(b) => (b.len() as u32) * 8,
        }
    }

    pub(crate) fn count_ones(&self) -> u32 {
        match &self.0 {
            Repr::Stack(v) => v.count_ones(),
            Repr::Heap(bytes) => bytes.iter().map(|b| b.count_ones()).sum(),
        }
    }

    pub(crate) fn fmt_bits(&self, f: &mut fmt::Formatter<'_>, width: u32) -> fmt::Result {
        for i in (0..width as usize).rev() {
            f.write_str(if self.get_bit(i) { "1" } else { "0" })?;
        }
        Ok(())
    }

    /// Whether any bit is set.
    pub(crate) fn any(&self) -> bool {
        match &self.0 {
            Repr::Stack(v) => *v != 0,
            Repr::Heap(bytes) => bytes.iter().any(|&b| b != 0),
        }
    }

    fn storage_byte(&self, i: usize) -> u8 {
        match &self.0 {
            Repr::Stack(v) => {
                if i < STACK_BYTES {
                    v.to_le_bytes()[i]
                } else {
                    0
                }
            }
            Repr::Heap(bytes) => bytes.get(i).copied().unwrap_or(0),
        }
    }

    fn storage_byte_count(&self) -> usize {
        match &self.0 {
            Repr::Stack(_) => STACK_BYTES,
            Repr::Heap(b) => b.len(),
        }
    }

    fn storage_bit_count(&self) -> usize {
        self.storage_byte_count() * 8
    }
}

impl From<u128> for BitReg {
    fn from(value: u128) -> Self {
        BitReg::stack(value)
    }
}

impl From<Box<[u8]>> for BitReg {
    fn from(bytes: Box<[u8]>) -> Self {
        BitReg::heap(bytes)
    }
}

pub(crate) struct BitIter<'a> {
    reg: &'a BitReg,
    pos: usize,
}

impl Iterator for BitIter<'_> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.pos >= self.reg.storage_bit_count() {
            return None;
        }
        let bit = self.reg.get_bit(self.pos);
        self.pos += 1;
        Some(bit)
    }
}

impl PartialEq for BitReg {
    fn eq(&self, other: &Self) -> bool {
        let n = self.storage_byte_count().max(other.storage_byte_count());
        for i in 0..n {
            if self.storage_byte(i) != other.storage_byte(i) {
                return false;
            }
        }
        true
    }
}

impl Eq for BitReg {}

/// Consistent with `==`: zero bytes past the last set bit do not count.
impl std::hash::Hash for BitReg {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let used = (0..self.storage_byte_count())
            .rev()
            .find(|&i| self.storage_byte(i) != 0)
            .map_or(0, |i| i + 1);
        for i in 0..used {
            self.storage_byte(i).hash(state);
        }
    }
}

/// Registers order as the unsigned integers their bits spell.
impl Ord for BitReg {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let n = self.storage_byte_count().max(other.storage_byte_count());
        let bytes = (0..n).rev();
        let mut orderings = bytes.map(|i| self.storage_byte(i).cmp(&other.storage_byte(i)));
        orderings
            .find(|ordering| ordering.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for BitReg {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq<u128> for BitReg {
    fn eq(&self, other: &u128) -> bool {
        self == &BitReg::stack(*other)
    }
}

impl PartialEq<BitReg> for u128 {
    fn eq(&self, other: &BitReg) -> bool {
        other == self
    }
}

fn binop_bytes(lhs: &BitReg, rhs: &BitReg, op: impl Fn(u8, u8) -> u8) -> BitReg {
    if let (Repr::Stack(a), Repr::Stack(b)) = (&lhs.0, &rhs.0) {
        let a_bytes = a.to_le_bytes();
        let b_bytes = b.to_le_bytes();
        let mut out = [0u8; STACK_BYTES];
        for i in 0..STACK_BYTES {
            out[i] = op(a_bytes[i], b_bytes[i]);
        }
        return BitReg::stack(u128::from_le_bytes(out));
    }
    let n = lhs.storage_byte_count().max(rhs.storage_byte_count());
    let mut out = vec![0u8; n];
    for (i, o) in out.iter_mut().enumerate() {
        *o = op(lhs.storage_byte(i), rhs.storage_byte(i));
    }
    BitReg::heap(out.into_boxed_slice())
}

impl BitAnd<&BitReg> for &BitReg {
    type Output = BitReg;
    fn bitand(self, rhs: &BitReg) -> BitReg {
        binop_bytes(self, rhs, |a, b| a & b)
    }
}

impl BitOr<&BitReg> for &BitReg {
    type Output = BitReg;
    fn bitor(self, rhs: &BitReg) -> BitReg {
        binop_bytes(self, rhs, |a, b| a | b)
    }
}

impl BitXor<&BitReg> for &BitReg {
    type Output = BitReg;
    fn bitxor(self, rhs: &BitReg) -> BitReg {
        binop_bytes(self, rhs, |a, b| a ^ b)
    }
}

impl Not for &BitReg {
    type Output = BitReg;
    fn not(self) -> BitReg {
        match &self.0 {
            Repr::Stack(v) => BitReg::stack(!v),
            Repr::Heap(bytes) => {
                let out: Vec<u8> = bytes.iter().map(|b| !b).collect();
                BitReg::heap(out.into_boxed_slice())
            }
        }
    }
}

impl Shl<u32> for &BitReg {
    type Output = BitReg;
    fn shl(self, amount: u32) -> BitReg {
        match &self.0 {
            Repr::Stack(v) => {
                if amount >= 128 {
                    BitReg::stack(0)
                } else {
                    BitReg::stack(v << amount)
                }
            }
            Repr::Heap(bytes) => {
                let n = bytes.len();
                let total_bits = n * 8;
                let mut out = vec![0u8; n];
                if (amount as usize) < total_bits {
                    let byte_shift = (amount / 8) as usize;
                    let bit_shift = amount % 8;
                    if bit_shift == 0 {
                        for i in byte_shift..n {
                            out[i] = bytes[i - byte_shift];
                        }
                    } else {
                        for i in byte_shift..n {
                            let lo = bytes[i - byte_shift] << bit_shift;
                            let hi = if i > byte_shift {
                                bytes[i - byte_shift - 1] >> (8 - bit_shift)
                            } else {
                                0
                            };
                            out[i] = lo | hi;
                        }
                    }
                }
                BitReg::heap(out.into_boxed_slice())
            }
        }
    }
}

impl Shr<u32> for &BitReg {
    type Output = BitReg;
    fn shr(self, amount: u32) -> BitReg {
        match &self.0 {
            Repr::Stack(v) => {
                if amount >= 128 {
                    BitReg::stack(0)
                } else {
                    BitReg::stack(v >> amount)
                }
            }
            Repr::Heap(bytes) => {
                let n = bytes.len();
                let total_bits = n * 8;
                let mut out = vec![0u8; n];
                if (amount as usize) < total_bits {
                    let byte_shift = (amount / 8) as usize;
                    let bit_shift = amount % 8;
                    if bit_shift == 0 {
                        for i in 0..(n - byte_shift) {
                            out[i] = bytes[i + byte_shift];
                        }
                    } else {
                        for i in 0..(n - byte_shift) {
                            let lo = bytes[i + byte_shift] >> bit_shift;
                            let hi = if i + byte_shift + 1 < n {
                                bytes[i + byte_shift + 1] << (8 - bit_shift)
                            } else {
                                0
                            };
                            out[i] = lo | hi;
                        }
                    }
                }
                BitReg::heap(out.into_boxed_slice())
            }
        }
    }
}

impl fmt::Display for BitReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_bits(f, self.storage_bit_count() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_picks_stack_for_small_widths() {
        assert!(matches!(BitReg::zeros(0).0, Repr::Stack(0)));
        assert!(matches!(BitReg::zeros(1).0, Repr::Stack(0)));
        assert!(matches!(BitReg::zeros(128).0, Repr::Stack(0)));
    }

    #[test]
    fn zeros_picks_heap_for_large_widths() {
        let r = BitReg::zeros(129);
        match r.0 {
            Repr::Heap(bytes) => assert_eq!(bytes.len(), 17),
            _ => panic!("expected heap"),
        }
        let r = BitReg::zeros(256);
        match r.0 {
            Repr::Heap(bytes) => assert_eq!(bytes.len(), 32),
            _ => panic!("expected heap"),
        }
    }

    #[test]
    fn from_u128_is_stack() {
        let r = BitReg::from(0xCAFE_u128);
        assert!(matches!(r.0, Repr::Stack(0xCAFE)));
    }

    #[test]
    fn from_box_is_heap() {
        let r = BitReg::from(Box::new([1u8, 2, 3]) as Box<[u8]>);
        match r.0 {
            Repr::Heap(b) => assert_eq!(&b[..], &[1, 2, 3]),
            _ => panic!("expected heap"),
        }
    }

    #[test]
    fn get_bit_stack() {
        let r = BitReg::stack(0b1010);
        assert!(!r.get_bit(0));
        assert!(r.get_bit(1));
        assert!(!r.get_bit(2));
        assert!(r.get_bit(3));
        assert!(!r.get_bit(127));
        assert!(!r.get_bit(200));
    }

    #[test]
    fn get_bit_heap() {
        let r = BitReg::heap(Box::new([0b1010_0000, 0b0000_0001]));
        assert!(!r.get_bit(0));
        assert!(r.get_bit(5));
        assert!(r.get_bit(7));
        assert!(r.get_bit(8));
        assert!(!r.get_bit(20));
    }

    #[test]
    fn set_bit_stack() {
        let mut r = BitReg::stack(0);
        r.set_bit(3, true);
        r.set_bit(127, true);
        assert!(matches!(r.0, Repr::Stack(v) if v == (1u128 << 3) | (1u128 << 127)));
        r.set_bit(3, false);
        assert!(matches!(r.0, Repr::Stack(v) if v == 1u128 << 127));
    }

    #[test]
    fn set_bit_heap() {
        let mut r = BitReg::zeros(200);
        r.set_bit(150, true);
        assert!(r.get_bit(150));
        assert!(!r.get_bit(149));
        r.set_bit(150, false);
        assert!(!r.get_bit(150));
    }

    #[test]
    fn stack_eq_heap_when_padding_is_zero() {
        let stack = BitReg::stack(0xAA_u128);
        let heap =
            BitReg::from(
                Box::new([0xAA, 0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) as Box<[u8]>,
            );
        assert_eq!(stack, heap);
    }

    #[test]
    fn stack_neq_heap_when_extra_bytes_nonzero() {
        let stack = BitReg::stack(0xAA_u128);
        let mut bytes = vec![0u8; 20];
        bytes[0] = 0xAA;
        bytes[17] = 0x01;
        let heap = BitReg::from(bytes.into_boxed_slice());
        assert_ne!(stack, heap);
    }

    #[test]
    fn round_trip_256_bit_heap() {
        let mut bytes = vec![0u8; 32];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        let r = BitReg::from(bytes.clone().into_boxed_slice());
        let bits: Vec<bool> = r.iter_bits().take(256).collect();
        let mut rebuilt = BitReg::zeros(256);
        for (i, &b) in bits.iter().enumerate() {
            rebuilt.set_bit(i, b);
        }
        assert_eq!(r, rebuilt);
    }

    #[test]
    fn resize_zeros_padding() {
        let r = BitReg::stack(0xFFFF_FFFF);
        let r = r.resize(12);
        match r.0 {
            Repr::Stack(v) => assert_eq!(v, 0xFFF),
            _ => panic!("expected stack"),
        }
    }

    #[test]
    fn resize_extends_to_heap() {
        let r = BitReg::stack(0xAA_u128);
        let r = r.resize(200);
        match &r.0 {
            Repr::Heap(b) => {
                assert_eq!(b.len(), 25);
                assert_eq!(b[0], 0xAA);
                for &byte in &b[1..] {
                    assert_eq!(byte, 0);
                }
            }
            _ => panic!("expected heap"),
        }
    }

    #[test]
    fn bitand_stack() {
        let a = BitReg::stack(0b1100);
        let b = BitReg::stack(0b1010);
        let r = &a & &b;
        assert!(matches!(r.0, Repr::Stack(0b1000)));
    }

    #[test]
    fn bitor_heap() {
        let a = BitReg::from(Box::new([0xF0_u8, 0x0F, 0x00]) as Box<[u8]>);
        let b = BitReg::from(Box::new([0x0F_u8, 0xF0, 0xFF]) as Box<[u8]>);
        let r = &a | &b;
        match r.0 {
            Repr::Heap(bytes) => assert_eq!(&bytes[..], &[0xFF, 0xFF, 0xFF]),
            _ => panic!("expected heap"),
        }
    }

    #[test]
    fn shl_heap_byte_aligned() {
        let r = BitReg::from(Box::new([0xFF_u8, 0x00, 0x00]) as Box<[u8]>);
        let s = &r << 8;
        match s.0 {
            Repr::Heap(b) => assert_eq!(&b[..], &[0x00, 0xFF, 0x00]),
            _ => panic!("expected heap"),
        }
    }

    #[test]
    fn shl_heap_cross_byte() {
        let r = BitReg::from(Box::new([0xFF_u8, 0x00]) as Box<[u8]>);
        let s = &r << 4;
        match s.0 {
            Repr::Heap(b) => assert_eq!(&b[..], &[0xF0, 0x0F]),
            _ => panic!("expected heap"),
        }
    }

    #[test]
    fn shr_heap_cross_byte() {
        let r = BitReg::from(Box::new([0x00_u8, 0xFF]) as Box<[u8]>);
        let s = &r >> 4;
        match s.0 {
            Repr::Heap(b) => assert_eq!(&b[..], &[0xF0, 0x0F]),
            _ => panic!("expected heap"),
        }
    }

    #[test]
    fn rotl_stack_full_width() {
        let mut r = BitReg::stack(0b0010_1010);
        r.rotl(0..8, 3);
        // 0b00101010 rotated left 3 = 0b01010001
        assert!(matches!(r.0, Repr::Stack(v) if v == 0b0101_0001));
    }

    #[test]
    fn rotl_heap_byte_spanning() {
        // 200-bit register, set bit 0, rotate left by 17 in 0..200.
        let mut r = BitReg::zeros(200);
        r.set_bit(0, true);
        r.rotl(0..200, 17);
        assert!(r.get_bit(17));
        assert!(!r.get_bit(0));
    }

    #[test]
    fn rotl_ignores_an_empty_or_reversed_slice() {
        let mut r = BitReg::from(0b1011_u128);
        #[allow(clippy::reversed_empty_ranges)]
        r.rotl(4..2, 1);
        r.rotl(3..3, 1);
        assert_eq!(r, 0b1011_u128);
    }

    #[test]
    fn rotl_by_the_complement_is_the_inverse() {
        let mut a = BitReg::zeros(200);
        for i in 0..200 {
            if i % 3 == 0 {
                a.set_bit(i, true);
            }
        }
        let original = a.clone();
        a.rotl(0..200, 17);
        a.rotl(0..200, 200 - 17);
        assert_eq!(a, original);
    }

    #[test]
    fn count_ones_works_for_both_variants() {
        let s = BitReg::stack(0b1011_1010);
        assert_eq!(s.count_ones(), 5);
        let h = BitReg::from(Box::new([0xFF_u8, 0x0F]) as Box<[u8]>);
        assert_eq!(h.count_ones(), 12);
    }
}
