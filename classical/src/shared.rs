use crate::error::{Error, Result};
use std::cell::RefCell;
use std::rc::Rc;

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
