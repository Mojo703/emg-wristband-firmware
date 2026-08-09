fn main() {
    // Only the device build has an ESP-IDF environment to report. A host build
    // is the tests in `src/phone.rs` and needs none of it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("espidf") {
        embuild::espidf::sysenv::output();
    }
}
