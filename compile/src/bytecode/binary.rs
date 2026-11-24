//! Binary serialization via postcard, with a magic-number prefix and
//! version byte so consumers can detect format incompatibilities.

use super::types::{Program, Version};

const MAGIC: [u8; 4] = *b"OQIB";

#[derive(Debug)]
pub enum DecodeError {
    BadMagic,
    IncompatibleVersion { found: Version, expected: Version },
    Postcard(postcard::Error),
}

impl From<postcard::Error> for DecodeError {
    fn from(e: postcard::Error) -> Self {
        DecodeError::Postcard(e)
    }
}

/// Encode a module to bytes. Output is `MAGIC | postcard(module)`.
pub fn to_bytes(module: &Program) -> Result<Vec<u8>, postcard::Error> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    let body = postcard::to_allocvec(module)?;
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a module from bytes. Rejects bad magic and incompatible
/// major versions.
pub fn from_bytes(bytes: &[u8]) -> Result<Program, DecodeError> {
    if bytes.len() < MAGIC.len() {
        return Err(DecodeError::BadMagic);
    }
    let (magic, rest) = bytes.split_at(MAGIC.len());
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let module: Program = postcard::from_bytes(rest)?;
    if module.version.major != Version::CURRENT.major {
        return Err(DecodeError::IncompatibleVersion {
            found: module.version,
            expected: Version::CURRENT,
        });
    }
    Ok(module)
}
