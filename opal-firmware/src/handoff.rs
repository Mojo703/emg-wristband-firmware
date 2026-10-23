use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

pub(crate) struct Handoff<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    capacity: usize,
}

impl<T> Clone for Handoff<T> {
    fn clone(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
            capacity: self.capacity,
        }
    }
}

impl<T> Handoff<T> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    pub(crate) fn offer(&self, item: T) -> Result<(), T> {
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        if queue.len() >= self.capacity {
            return Err(item);
        }
        queue.push_back(item);
        Ok(())
    }

    pub(crate) fn take(&self) -> Option<T> {
        self.queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_handoff_returns_the_item() {
        let handoff = Handoff::with_capacity(1);
        assert_eq!(handoff.offer(1), Ok(()));
        assert_eq!(handoff.offer(2), Err(2));
        assert_eq!(handoff.clone().take(), Some(1));
        assert_eq!(handoff.take(), None);
    }
}
