//! The bytecode file format.
//!
//! A file is the four bytes `OQIB`, the format version as a little-endian
//! `u16` ([`FORMAT_VERSION`]), then the [postcard] encoding of the
//! [`Program`]. The header is read before the body, so a file from another
//! format version is reported as such instead of as corrupt.
//!
//! Decoding is safe on untrusted bytes: after the body decodes, every index
//! a program holds (procedures, blocks, registers, constants, strings,
//! symbols, qubits) is checked before the program is returned (see
//! `validate`), so the VM cannot be sent out of bounds.
//!
//! [postcard]: https://docs.rs/postcard

use std::fmt;

use super::types::Program;
use super::validate;

/// The version of the bytecode format this build reads and writes. It
/// changes whenever the encoding of a [`Program`] does.
pub const FORMAT_VERSION: u16 = 1;

const MAGIC: [u8; 4] = *b"OQIB";
const HEADER_LEN: usize = MAGIC.len() + 2;

/// A [`Program`] could not be encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeError {
    message: String,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot encode the program: {}", self.message)
    }
}

impl std::error::Error for EncodeError {}

/// Bytes could not be decoded as a [`Program`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// The bytes are not bytecode: they don't start with `OQIB`.
    NotBytecode,
    /// Bytecode of a format version this build doesn't read.
    UnsupportedVersion {
        /// The version the bytes declare.
        found: u16,
        /// The version this build reads ([`FORMAT_VERSION`]).
        supported: u16,
    },
    /// The body is truncated or corrupt.
    Malformed(String),
    /// The body decodes, but describes an inconsistent program (an index
    /// out of range, a block that doesn't exist, …).
    Invalid(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::NotBytecode => f.write_str("not oqi bytecode (no `OQIB` header)"),
            DecodeError::UnsupportedVersion { found, supported } => write!(
                f,
                "bytecode format version {found} is not supported (this build reads version {supported})"
            ),
            DecodeError::Malformed(why) => write!(f, "malformed bytecode: {why}"),
            DecodeError::Invalid(why) => write!(f, "invalid bytecode: {why}"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl Program {
    /// Encode the program in the bytecode format; [`Program::from_bytes`]
    /// reads it back.
    ///
    /// # Errors
    ///
    /// An [`EncodeError`] if a value in the program can't be serialized.
    pub fn to_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        postcard::to_extend(self, out).map_err(|e| EncodeError {
            message: e.to_string(),
        })
    }

    /// Decode a program written by [`Program::to_bytes`].
    ///
    /// # Errors
    ///
    /// A [`DecodeError`] if the bytes are not bytecode, are of another
    /// format version, are truncated or corrupt, or describe an
    /// inconsistent program.
    pub fn from_bytes(bytes: &[u8]) -> Result<Program, DecodeError> {
        let version = header_version(bytes).ok_or(DecodeError::NotBytecode)?;
        if version != FORMAT_VERSION {
            return Err(DecodeError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let (program, rest): (Program, &[u8]) = postcard::take_from_bytes(&bytes[HEADER_LEN..])
            .map_err(|e| DecodeError::Malformed(e.to_string()))?;
        if !rest.is_empty() {
            return Err(DecodeError::Malformed(format!(
                "{} unexpected bytes after the program",
                rest.len()
            )));
        }
        validate::validate(program).map_err(DecodeError::Invalid)
    }

    /// Whether `bytes` start with the bytecode header (of any format
    /// version): how to tell bytecode from source text.
    pub fn is_bytecode(bytes: &[u8]) -> bool {
        bytes.starts_with(&MAGIC)
    }
}

/// The format version in the header of `bytes`, if they have one.
fn header_version(bytes: &[u8]) -> Option<u16> {
    let header = bytes.get(..HEADER_LEN)?;
    let (magic, version) = header.split_at(MAGIC.len());
    (magic == MAGIC).then(|| u16::from_le_bytes([version[0], version[1]]))
}
