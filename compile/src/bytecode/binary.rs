//! Binary serialization via postcard, with a magic-number prefix and
//! version byte so consumers can detect format incompatibilities.

use super::types::{BcModule, BcVersion};

const MAGIC: [u8; 4] = *b"OQIB";

#[derive(Debug)]
pub enum EncodeError {
    Postcard(postcard::Error),
}

impl From<postcard::Error> for EncodeError {
    fn from(e: postcard::Error) -> Self {
        EncodeError::Postcard(e)
    }
}

#[derive(Debug)]
pub enum DecodeError {
    BadMagic,
    IncompatibleVersion { found: BcVersion, expected: BcVersion },
    Postcard(postcard::Error),
}

impl From<postcard::Error> for DecodeError {
    fn from(e: postcard::Error) -> Self {
        DecodeError::Postcard(e)
    }
}

/// Encode a module to bytes. Output is `MAGIC | postcard(module)`.
pub fn to_bytes(module: &BcModule) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    let body = postcard::to_allocvec(module)?;
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a module from bytes. Rejects bad magic and incompatible
/// major versions.
pub fn from_bytes(bytes: &[u8]) -> Result<BcModule, DecodeError> {
    if bytes.len() < MAGIC.len() {
        return Err(DecodeError::BadMagic);
    }
    let (magic, rest) = bytes.split_at(MAGIC.len());
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let module: BcModule = postcard::from_bytes(rest)?;
    if module.version.major != BcVersion::CURRENT.major {
        return Err(DecodeError::IncompatibleVersion {
            found: module.version,
            expected: BcVersion::CURRENT,
        });
    }
    Ok(module)
}
