//! Pure ordering guards at the calibration-flow adapter boundary.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActionGuard<S> {
    in_flight: Option<S>,
}

impl<S> Default for ActionGuard<S> {
    fn default() -> Self {
        Self { in_flight: None }
    }
}

impl<S: Copy + PartialEq> ActionGuard<S> {
    pub(crate) fn is_ready(&self) -> bool {
        self.in_flight.is_none()
    }

    pub(crate) fn dispatch(&mut self, step: S) -> Result<(), S> {
        if let Some(in_flight) = self.in_flight {
            return Err(in_flight);
        }
        self.in_flight = Some(step);
        Ok(())
    }

    pub(crate) fn complete(&mut self, reported: S) -> Result<(), Option<S>> {
        if self.in_flight != Some(reported) {
            return Err(self.in_flight);
        }
        self.in_flight = None;
        Ok(())
    }

    pub(crate) fn cancel(&mut self) {
        self.in_flight = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlushedRows(usize);

impl FlushedRows {
    pub(crate) fn count(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BufferedRows(pub(crate) usize);

pub(crate) fn rows_ready_to_install(
    buffered: usize,
    flushed: usize,
) -> Result<FlushedRows, BufferedRows> {
    if buffered == 0 {
        Ok(FlushedRows(flushed))
    } else {
        Err(BufferedRows(buffered))
    }
}

#[cfg(test)]
mod tests {
    use super::{rows_ready_to_install, ActionGuard, BufferedRows};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Step {
        Flush,
        Fit,
    }

    #[test]
    fn an_in_flight_action_blocks_repeated_dispatch_until_its_report() {
        let mut guard = ActionGuard::default();

        assert_eq!(guard.dispatch(Step::Flush), Ok(()));
        assert_eq!(guard.dispatch(Step::Flush), Err(Step::Flush));
        assert_eq!(guard.dispatch(Step::Fit), Err(Step::Flush));
        assert!(!guard.is_ready());

        assert_eq!(guard.complete(Step::Flush), Ok(()));
        assert!(guard.is_ready());
        assert_eq!(guard.dispatch(Step::Fit), Ok(()));
    }

    #[test]
    fn a_report_for_the_wrong_action_does_not_release_the_guard() {
        let mut guard = ActionGuard::default();
        guard.dispatch(Step::Flush).unwrap();

        assert_eq!(guard.complete(Step::Fit), Err(Some(Step::Flush)));
        assert_eq!(guard.dispatch(Step::Fit), Err(Step::Flush));
        guard.cancel();
        assert_eq!(guard.dispatch(Step::Fit), Ok(()));
    }

    #[test]
    fn buffered_rows_cannot_be_present_at_install() {
        assert_eq!(rows_ready_to_install(3, 45), Err(BufferedRows(3)));
        assert_eq!(rows_ready_to_install(0, 45).unwrap().count(), 45);
    }
}
