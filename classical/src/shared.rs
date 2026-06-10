use crate::error::{Error, Result};
use std::cell::RefCell;
use std::rc::Rc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug)]
pub struct Shared<T>(Rc<RefCell<T>>);

pub type Ref<'a, T> = std::cell::Ref<'a, T>;
pub type RefMut<'a, T> = std::cell::RefMut<'a, T>;

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }

    pub fn borrow(&self) -> Result<Ref<'_, T>> {
        self.0.try_borrow().map_err(|_| Error::BadBorrow)
    }

    pub fn borrow_mut(&self) -> Result<RefMut<'_, T>> {
        self.0.try_borrow_mut().map_err(|_| Error::BadBorrow)
    }
}

// `Shared<T>` serializes as just the inner `T`. Sharing identity is
// not preserved across the serialization boundary — on deserialize a
// fresh `Shared::new(...)` is constructed.
impl<T: Serialize> Serialize for Shared<T> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.borrow()
            .map_err(|_| serde::ser::Error::custom("Shared<T> already mutably borrowed"))?
            .serialize(s)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Shared<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Ok(Shared::new(T::deserialize(d)?))
    }
}
