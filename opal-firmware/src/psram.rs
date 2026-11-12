//! Fixed-capacity buffers in the XIAO's external PSRAM.
//!
//! `sdkconfig.defaults` turns PSRAM on in caps-allocation mode, so nothing
//! reaches it unless asked for by capability. This is the asking: a buffer of
//! plain `Copy` values, allocated once, filled from the front, never grown.

use core::marker::PhantomData;
use core::ptr::NonNull;

use esp_idf_svc::sys;

pub(crate) struct PsramVec<T: Copy> {
    start: NonNull<T>,
    capacity: usize,
    len: usize,
    _owns: PhantomData<T>,
}

// SAFETY: the buffer is owned exclusively, like a `Vec<T>`.
unsafe impl<T: Copy + Send> Send for PsramVec<T> {}

impl<T: Copy> PsramVec<T> {
    /// `None` when the board has no PSRAM or not enough of it free.
    pub fn with_capacity(capacity: usize) -> Option<Self> {
        let bytes = capacity.checked_mul(core::mem::size_of::<T>())?;
        // SAFETY: plain allocation; the result is checked before use.
        let raw = unsafe { sys::heap_caps_malloc(bytes, sys::MALLOC_CAP_SPIRAM) };
        let start = NonNull::new(raw.cast::<T>())?;
        if raw as usize % core::mem::align_of::<T>() != 0 {
            // SAFETY: freeing what heap_caps_malloc returned.
            unsafe { sys::heap_caps_free(raw) };
            return None;
        }
        Some(Self {
            start,
            capacity,
            len: 0,
            _owns: PhantomData,
        })
    }

    /// `false` when full.
    pub fn push(&mut self, value: T) -> bool {
        if self.len == self.capacity {
            return false;
        }
        // SAFETY: `len < capacity`, inside the allocation.
        unsafe { self.start.as_ptr().add(self.len).write(value) };
        self.len += 1;
        true
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the first `len` values were written by `push`.
        unsafe { core::slice::from_raw_parts(self.start.as_ptr(), self.len) }
    }

    /// The whole capacity, written or not, for filling in place. Values past
    /// `len` are whatever the allocation held.
    pub fn spare_capacity(&mut self) -> &mut [core::mem::MaybeUninit<T>] {
        // SAFETY: the allocation holds `capacity` values of `T`.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.start.as_ptr().cast::<core::mem::MaybeUninit<T>>(),
                self.capacity,
            )
        }
    }

    /// Declare the first `len` values written.
    ///
    /// # Safety
    /// Every value below `len` must have been initialized.
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len.min(self.capacity);
    }
}

impl<T: Copy> Drop for PsramVec<T> {
    fn drop(&mut self) {
        // SAFETY: freeing what heap_caps_malloc returned, once.
        unsafe { sys::heap_caps_free(self.start.as_ptr().cast()) };
    }
}
