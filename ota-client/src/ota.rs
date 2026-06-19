//! OTA update flow: download an image over HTTP and write it to the inactive
//! app slot, then mark it as the next boot partition.
//!
//! The actual flash partitioning and boot-slot switching is handled by the
//! ESP-IDF OTA subsystem (`EspOta`); we just stream bytes into it.

use anyhow::{bail, Result};
use embedded_svc::io::Read;
use esp_idf_svc::http::client::{Configuration as HttpConfig, EspHttpConnection};
use esp_idf_svc::http::Method;
use esp_idf_svc::ota::EspOta;
use log::info;

const BUF_SIZE: usize = 4096;

pub(crate) struct Ota {
    url: &'static str,
}

impl Ota {
    pub(crate) fn new(url: &'static str) -> Self {
        Self { url }
    }

    /// Download the firmware at the predefined `url` and stage it as the next boot image.
    ///
    /// On success the inactive OTA slot holds the new image and is set as the boot
    /// partition; the caller should reboot. On any failure the partially written
    /// update is aborted and the current firmware stays active.
    pub(crate) fn run_update(&self) -> Result<()> {
        info!("OTA: fetching {}", self.url);

        let mut conn = EspHttpConnection::new(&HttpConfig {
            buffer_size: Some(BUF_SIZE),
            ..Default::default()
        })?;

        conn.initiate_request(Method::Get, self.url, &[])?;
        conn.initiate_response()?;

        let status = conn.status();
        if status != 200 {
            bail!("server returned HTTP {status}");
        }

        let mut ota = EspOta::new()?;
        let mut update = ota.initiate_update()?;

        let mut buf = [0u8; BUF_SIZE];
        let mut total: usize = 0;
        loop {
            let n = Read::read(&mut conn, &mut buf)?;
            if n == 0 {
                break;
            }
            if let Err(e) = update.write(&buf[..n]) {
                update.abort()?;
                bail!("flash write failed after {total} bytes: {e:?}");
            }
            total += n;
        }

        if total == 0 {
            update.abort()?;
            bail!("downloaded 0 bytes; nothing written");
        }

        info!("OTA: downloaded {total} bytes, finalizing");
        update.complete()?;
        info!("OTA: image staged and set as boot partition");
        Ok(())
    }
}
