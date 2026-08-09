// The esp-idf sysenv output is what links the bench example against the IDF.
// The driver itself is `no_std` over embedded-hal and needs none of it, and
// with `default-features = false` there is no IDF in the build at all — so
// calling into it would fail the build of every host consumer of this crate.
//
// `feedback-vocabulary` is one such consumer: it borrows the sequencer types to
// describe haptic patterns and runs its collision tests on a laptop.
fn main() {
    if std::env::var_os("CARGO_FEATURE_BENCH").is_some() {
        embuild::espidf::sysenv::output();
    }
}
