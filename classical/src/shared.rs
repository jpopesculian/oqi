use crate::error::{Error, Result};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Shared<T>(Rc<RefCell<T>>);

/// Identity: two handles are equal when they share one value.
impl<T> PartialEq for Shared<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

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
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Shared<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.borrow()
            .map_err(|_| serde::ser::Error::custom("Shared<T> already mutably borrowed"))?
            .serialize(s)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Shared<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Ok(Shared::new(T::deserialize(d)?))
    }
}
