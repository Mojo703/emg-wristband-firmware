//! The frame read, moved into the DRDY interrupt.
//!
//! The ADS1298 has no FIFO. A conversion that is not clocked out inside one
//! ~487 µs sample period is gone, and a read still clocking when the next
//! conversion lands reloads the shift register mid-readout and returns a frame
//! that passes the status marker while carrying spliced data. Both failures are
//! deadline failures, so the read runs where the deadline is not negotiable:
//! an IRAM interrupt handler that starts within microseconds of the edge and,
//! measured on the bench, clocks the frame's last bit 32 µs after it — against a
//! ~487 µs period, and whatever the scheduler, the radio, or a
//! flash-cache-disabled window is doing.
//!
//! # One edge, both chips
//!
//! The chips share SCLK and convert on one clock from one START edge, so a single
//! interrupt on chip 0's DRDY reads chip 0 and then chip 1, back to back, under
//! one timestamp. Chip 1's DRDY cannot carry its own read: a rising SCLK pulls
//! DRDY high regardless of CS (SBAS459K §9.4.1.2), and its conversion is already
//! complete by the time chip 0's read ends. Its falling edges are only counted, so
//! the pipeline can tell when chip 1 stops converting while chip 0 carries on.
//!
//! # Bus ownership
//!
//! The shared SPI host has exactly one owner at any instant, and the handover
//! is explicit:
//!
//! - The esp-idf SPI master driver owns the host during bring-up, the
//!   configuration readback, and warm recovery. It is not ISR-safe and must
//!   never run while this module's interrupt can fire.
//! - This module owns the host while the DRDY interrupt is enabled. It
//!   drives the GPSPI registers directly, taking no driver lock and making no
//!   driver call.
//!
//! [`FrameReader::disable`] closes the interrupt window and waits past one
//! whole frame read, so the driver can only start once no handler is in
//! flight; [`FrameReader::enable`] reopens it. Warm recovery brackets its
//! driver traffic with exactly that pair.
//!
//! The two owners cannot corrupt each other's register state either. The
//! [`Recipe`] is captured from the driver's own configuration of the host for
//! this exact transfer (see [`FrameReader::claim`]), and the handler rewrites
//! every register of it before every frame, so command-path traffic at a
//! different clock and length leaves nothing behind. In the other direction the
//! driver clears `trans_done` and asserts the host is idle at the start of each
//! of its transactions, and it keeps its own completion interrupt disabled at
//! the interrupt controller whenever no transaction is queued — so a raw
//! transaction's completion can neither dispatch the driver's ISR nor confuse
//! its next transaction.
//!
//! # What crosses out of interrupt context
//!
//! One lock-free single-producer single-consumer ring per chip, in DRAM,
//! power-of-two sized, published through an atomic index. The handler writes
//! raw FIFO words and a device-clock stamp; every decode, status check, health
//! decision, and telemetry read stays on [`super::acquisition::pipeline`]'s thread.

use anyhow::{anyhow, Result};
use esp_idf_svc::sys::{
    esp, esp_rom_delay_us, esp_timer_get_time, gpio_int_type_t_GPIO_INTR_NEGEDGE,
    gpio_intr_disable, gpio_intr_enable, gpio_isr_handler_add, gpio_set_intr_type,
    spi_device_acquire_bus, spi_device_handle_t, spi_device_release_bus,
    spi_host_device_t_SPI2_HOST, spi_host_device_t_SPI3_HOST, DR_REG_GPIO_BASE, DR_REG_SPI2_BASE,
    DR_REG_SPI3_BASE,
};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, Ordering};

use super::channel::DEVICE_COUNT;
use super::decode::FRAME_BYTES;

/// FIFO words one frame occupies: 27 bytes rounded up to whole 32-bit words.
/// The last word carries three payload bytes and one byte of clocked-out
/// nothing, which [`words_to_bytes`] drops.
const FRAME_WORDS: usize = FRAME_BYTES.div_ceil(4);

/// Frames one chip's ring holds. At the ~2 kHz conversion rate this is ~32 ms
/// of slack against the pipeline thread, whose drain the bench measures falling
/// at most ~3 ms behind. The margin is the point: the ring is what lets a flash write, a
/// wifi burst, or any other multi-millisecond hold on the thread cost latency
/// instead of conversions. Power of two, so the index arithmetic is a mask.
///
/// It is a boot-time static, not an allocation: nothing on this path may touch
/// the heap, and the handler may run with the flash cache disabled, so the
/// slots have to be in DRAM.
const RING_CAPACITY: usize = 64;
const RING_MASK: u32 = RING_CAPACITY as u32 - 1;

/// Word offsets of the GPSPI registers this module touches, from the host's
/// register base (`spi_dev_t` in soc/spi_struct.h).
///
/// These are `const`, so they compile into instruction immediates rather than a
/// flash-resident table the handler would have to load.
const CMD: usize = 0;
const CTRL: usize = 2;
const CLOCK: usize = 3;
const USER: usize = 4;
const USER1: usize = 5;
const USER2: usize = 6;
const MS_DLEN: usize = 7;
const MISC: usize = 8;
const DIN_MODE: usize = 9;
const DIN_NUM: usize = 10;
const DOUT_MODE: usize = 11;
const DMA_CONF: usize = 12;
const DMA_INT_ENA: usize = 13;
const DMA_INT_CLR: usize = 14;
const DMA_INT_SET: usize = 17;
const DATA_BUF: usize = 38;
const SLAVE: usize = 56;
const CLK_GATE: usize = 58;

/// `SPI_CMD_REG`'s two command bits. `UPDATE` synchronises the configuration
/// registers into the SPI clock domain and self-clears; `USR` starts the
/// transfer and self-clears when it completes.
const CMD_UPDATE: u32 = 1 << 23;
const CMD_USR: u32 = 1 << 24;
const TRANS_DONE_INT: u32 = 1 << 12;

const OWNER_DRIVER: u32 = 0;
const OWNER_RAW: u32 = 1;
const HANDLER_GATE_OPEN: u32 = 1;
const HANDLER_ACTIVE_INCREMENT: u32 = 2;
const HANDOFF_TIMEOUT_US: i64 = 10_000;

/// Spins a register poll is allowed before the handler gives up and counts a
/// read fault. A 27-byte transfer at 8 MHz completes in ~27 µs; this bound is
/// roughly an order of magnitude past that, so it can only be reached by a host
/// that has stopped responding — and reaching it must return control rather
/// than wedge the interrupt.
const POLL_SPIN_LIMIT: u32 = 10_000;

/// The keep-out around a DRDY pulse: 4 tCLK ≈ 2 µs at fCLK = 2.048 MHz, during
/// which no SCLK may be presented (SBAS459K §9.5.2.6, tUPDATE). Interrupt entry
/// latency alone covers it; the wait is here so the guarantee is structural
/// rather than a property of whatever the interrupt controller happened to cost
/// that cycle.
const UPDATE_KEEP_OUT_MICROSECONDS: u32 = 2;

/// tSCCS: 4 tCLK must pass after the last SCLK before CS may rise
/// (SBAS459K §9.5.1.1). Same 2 µs, same reason to spend it explicitly.
const CHIP_SELECT_HOLD_MICROSECONDS: u32 = 2;

/// One frame as the interrupt captured it: the device clock at the DRDY edge,
/// how long the transfer itself took, and the FIFO words verbatim. Decoding is
/// the reader thread's job.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct FrameSlot {
    edge_us: u64,
    read_us: u32,
    words: [u32; FRAME_WORDS],
}

impl FrameSlot {
    const EMPTY: Self = Self {
        edge_us: 0,
        read_us: 0,
        words: [0; FRAME_WORDS],
    };
}

/// One chip's frames crossing out of interrupt context. Both rings fill from the
/// same handler.
///
/// Single producer (the handler), single consumer (the pipeline thread). The
/// slot contents are ordinary memory; `write_index` is what publishes them, so
/// the producer's release store and the consumer's acquire load are the whole
/// synchronisation.
struct FrameRing {
    slots: UnsafeCell<[FrameSlot; RING_CAPACITY]>,
    write_index: AtomicU32,
    read_index: AtomicU32,
    /// Frames the handler read and had nowhere to put. Impossible while the
    /// pipeline thread drains at its priority — the ring is ~32 ms deep — and
    /// counted anyway, because the alternative to counting it is a silent hole.
    overruns: AtomicU32,
    /// Transfers that hit [`POLL_SPIN_LIMIT`] instead of completing.
    read_faults: AtomicU32,
    /// DRDY edges the handler serviced, cumulative. This is the honest
    /// "serviced edges" number: it counts the conversions actually clocked out
    /// of the chip, not the wakes some thread got around to.
    edges: AtomicU32,
}

// The `UnsafeCell` is disciplined by the index protocol above, not by a lock.
unsafe impl Sync for FrameRing {}

impl FrameRing {
    const fn new() -> Self {
        Self {
            slots: UnsafeCell::new([FrameSlot::EMPTY; RING_CAPACITY]),
            write_index: AtomicU32::new(0),
            read_index: AtomicU32::new(0),
            overruns: AtomicU32::new(0),
            read_faults: AtomicU32::new(0),
            edges: AtomicU32::new(0),
        }
    }
}

/// Every GPSPI register the transfer depends on, captured from the driver's own
/// configuration of the host.
///
/// Deriving the recipe instead of writing it out is deliberate: the driver
/// already knows how to express mode 1 at 8 MHz, full duplex, MSB first, 27
/// bytes, no DMA, software chip select — on this silicon, at this clock source,
/// with the timing compensation its own tables picked. Copying its answer
/// cannot disagree with the driver path the bench validated, and cannot drift
/// when either side is reconfigured.
#[derive(Clone, Copy)]
struct Recipe {
    ctrl: u32,
    clock: u32,
    user: u32,
    user1: u32,
    user2: u32,
    ms_dlen: u32,
    misc: u32,
    din_mode: u32,
    din_num: u32,
    dout_mode: u32,
    dma_conf: u32,
    slave: u32,
    clk_gate: u32,
}

impl Recipe {
    const EMPTY: Self = Self {
        ctrl: 0,
        clock: 0,
        user: 0,
        user1: 0,
        user2: 0,
        ms_dlen: 0,
        misc: 0,
        din_mode: 0,
        din_num: 0,
        dout_mode: 0,
        dma_conf: 0,
        slave: 0,
        clk_gate: 0,
    };
}

/// Everything the handler needs, as plain integers so the whole thing lives in
/// DRAM and carries no pointer that could outlive what it points at. Written
/// once, from task context, while the interrupt is disabled.
#[derive(Clone, Copy)]
struct HostReader {
    /// The shared GPSPI host's register base.
    spi_base: u32,
    recipe: Recipe,
    chip_selects: [ChipSelect; DEVICE_COUNT],
}

/// The `GPIO_OUT*_W1TS`/`W1TC` pair and bit for one chip's chip select. Driving
/// CS from its registers rather than through `PinDriver` keeps the handler clear
/// of esp-idf-hal, which is not IRAM-resident.
#[derive(Clone, Copy)]
struct ChipSelect {
    set: u32,
    clear: u32,
    mask: u32,
}

impl ChipSelect {
    const EMPTY: Self = Self {
        set: 0,
        clear: 0,
        mask: 0,
    };

    fn for_pin(pin: u8) -> Self {
        // GPIO 0..31 live in `GPIO_OUT_W1TS/W1TC` at +0x08/+0x0C, 32..48 in the
        // `GPIO_OUT1_*` pair at +0x14/+0x18.
        let (set_offset, clear_offset, bit) = if pin < 32 {
            (0x08, 0x0C, pin)
        } else {
            (0x14, 0x18, pin - 32)
        };
        Self {
            set: DR_REG_GPIO_BASE + set_offset,
            clear: DR_REG_GPIO_BASE + clear_offset,
            mask: 1 << bit,
        }
    }
}

impl HostReader {
    const EMPTY: Self = Self {
        spi_base: 0,
        recipe: Recipe::EMPTY,
        chip_selects: [ChipSelect::EMPTY; DEVICE_COUNT],
    };
}

/// A static whose interior mutation is disciplined by the enable/disable window
/// rather than by a lock: writers run in task context with the interrupt off.
struct InterruptShared<T>(UnsafeCell<T>);
unsafe impl<T> Sync for InterruptShared<T> {}

/// Handler state, in `.bss` so the handler reaches it with the flash cache
/// disabled and allocates nothing to do so.
static READER: InterruptShared<HostReader> = InterruptShared(UnsafeCell::new(HostReader::EMPTY));
static RINGS: [FrameRing; DEVICE_COUNT] = [FrameRing::new(), FrameRing::new()];
static OWNER: AtomicU32 = AtomicU32::new(OWNER_DRIVER);
static HANDLER_GATE: AtomicU32 = AtomicU32::new(0);
/// Falling edges on chip 1's DRDY, which the handler does not read on.
static COUNTED_EDGES: AtomicU32 = AtomicU32::new(0);

/// Writes one 32-bit peripheral register.
///
/// # Safety
///
/// `base` must be a GPSPI register base and `word` an offset inside it.
#[inline(always)]
unsafe fn write_register(base: u32, word: usize, value: u32) {
    std::ptr::write_volatile((base as *mut u32).add(word), value);
}

/// Reads one 32-bit peripheral register.
///
/// # Safety
///
/// As [`write_register`].
#[inline(always)]
unsafe fn read_register(base: u32, word: usize) -> u32 {
    std::ptr::read_volatile((base as *const u32).add(word))
}

/// Clocks one frame out of each chip, in chip order, and publishes each to that
/// chip's ring under one edge stamp.
///
/// Runs in interrupt context and from [`FrameReader::validate`] in task
/// context; the two never overlap, because validation happens before the
/// interrupt is ever enabled. Everything it touches is IRAM code, DRAM data, or
/// a peripheral register: no allocation, no lock, no FreeRTOS call, no flash.
///
/// # Safety
///
/// The host must have been claimed and be owned by this module (see the
/// module's ownership contract).
#[link_section = ".iram1.adc_read_frames"]
unsafe fn read_frames_into_rings() {
    let edge_us = esp_timer_get_time() as u64;
    for index in 0..DEVICE_COUNT {
        read_frame_into_ring(index, edge_us);
    }
}

/// Clocks one frame out of chip `index` into its ring. `edge_us` is the DRDY
/// edge both chips' frames are stamped with; `read_us` is measured from it, so
/// chip 1's includes chip 0's read.
///
/// # Safety
///
/// As [`read_frames_into_rings`].
#[link_section = ".iram1.adc_read_frame"]
unsafe fn read_frame_into_ring(index: usize, edge_us: u64) {
    let reader = &*READER.0.get();
    let chip_select = *reader.chip_selects.get_unchecked(index);
    let ring = RINGS.get_unchecked(index);
    let base = reader.spi_base;

    // CS low for the whole frame (SBAS459K §9.5.1.1), then the tUPDATE keep-out
    // before the first SCLK.
    std::ptr::write_volatile(chip_select.clear as *mut u32, chip_select.mask);
    esp_rom_delay_us(UPDATE_KEEP_OUT_MICROSECONDS);

    let recipe = &reader.recipe;
    write_register(base, CTRL, recipe.ctrl);
    write_register(base, CLOCK, recipe.clock);
    write_register(base, USER, recipe.user);
    write_register(base, USER1, recipe.user1);
    write_register(base, USER2, recipe.user2);
    write_register(base, MS_DLEN, recipe.ms_dlen);
    write_register(base, MISC, recipe.misc);
    write_register(base, DIN_MODE, recipe.din_mode);
    write_register(base, DIN_NUM, recipe.din_num);
    write_register(base, DOUT_MODE, recipe.dout_mode);
    write_register(base, DMA_CONF, recipe.dma_conf);
    write_register(base, SLAVE, recipe.slave);
    write_register(base, CLK_GATE, recipe.clk_gate);

    // DIN must stay low for all 216 clocks (SBAS459K §9.4.1.3): the command
    // decoder listens during RDATAC, so a floating or stale MOSI line is a
    // potential opcode. The transfer is full duplex against these zeros, and
    // the received bytes land back in the same words.
    for word in 0..FRAME_WORDS {
        write_register(base, DATA_BUF + word, 0);
    }

    let mut faulted = false;
    let command = read_register(base, CMD);
    write_register(base, CMD, command | CMD_UPDATE);
    let mut spins = 0;
    while read_register(base, CMD) & CMD_UPDATE != 0 {
        spins += 1;
        if spins >= POLL_SPIN_LIMIT {
            faulted = true;
            break;
        }
    }
    if !faulted {
        let command = read_register(base, CMD);
        write_register(base, CMD, command | CMD_USR);
        let mut spins = 0;
        while read_register(base, CMD) & CMD_USR != 0 {
            spins += 1;
            if spins >= POLL_SPIN_LIMIT {
                faulted = true;
                break;
            }
        }
    }
    let read_us = (esp_timer_get_time() as u64).saturating_sub(edge_us) as u32;

    // tSCCS, then the decoder-reset edge. The FIFO is read after CS rises: the
    // words are latched, DOUT going high-impedance does not touch them, and
    // holding CS low for the copy would only lengthen the frame.
    esp_rom_delay_us(CHIP_SELECT_HOLD_MICROSECONDS);
    std::ptr::write_volatile(chip_select.set as *mut u32, chip_select.mask);

    if faulted {
        ring.read_faults.fetch_add(1, Ordering::Relaxed);
        ring.edges.fetch_add(1, Ordering::Relaxed);
        return;
    }

    let write = ring.write_index.load(Ordering::Relaxed);
    let read = ring.read_index.load(Ordering::Acquire);
    if write.wrapping_sub(read) >= RING_CAPACITY as u32 {
        ring.overruns.fetch_add(1, Ordering::Relaxed);
    } else {
        let slot = (*ring.slots.get()).get_unchecked_mut((write & RING_MASK) as usize);
        slot.edge_us = edge_us;
        slot.read_us = read_us;
        for word in 0..FRAME_WORDS {
            *slot.words.get_unchecked_mut(word) = read_register(base, DATA_BUF + word);
        }
        ring.write_index
            .store(write.wrapping_add(1), Ordering::Release);
    }
    ring.edges.fetch_add(1, Ordering::Relaxed);
}

/// The handler on chip 0's DRDY: reads both chips.
#[link_section = ".iram1.adc_data_ready"]
unsafe extern "C" fn data_ready_handler(_argument: *mut core::ffi::c_void) {
    let gate = &HANDLER_GATE;
    let mut state = gate.load(Ordering::Acquire);
    loop {
        if state & HANDLER_GATE_OPEN == 0 {
            return;
        }
        match gate.compare_exchange_weak(
            state,
            state + HANDLER_ACTIVE_INCREMENT,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break,
            Err(current) => state = current,
        }
    }
    read_frames_into_rings();
    gate.fetch_sub(HANDLER_ACTIVE_INCREMENT, Ordering::Release);
}

/// The handler on chip 1's DRDY: counts the edge and touches nothing else.
#[link_section = ".iram1.adc_counted_edge"]
unsafe extern "C" fn counted_edge_handler(_argument: *mut core::ffi::c_void) {
    COUNTED_EDGES.fetch_add(1, Ordering::Relaxed);
}

/// Installs the GPIO interrupt dispatcher the DRDY handlers hang off.
///
/// Called instead of esp-idf-hal's `enable_isr_service`, for the two flags that
/// matter here. `ESP_INTR_FLAG_IRAM` keeps the dispatcher and its handlers
/// running while the flash cache is disabled — an NVS write must not cost
/// conversions. Level 3 puts DRDY above the level-1 peripheral interrupts, so
/// the edge is serviced even mid-burst; if the interrupt allocator cannot find
/// a level-3 slot, level 1 with IRAM is still the point of the exercise, so the
/// install falls back rather than failing bring-up.
///
/// Where this runs decides where every DRDY read runs: esp-idf allocates the
/// interrupt on the calling core. See [`crate::cores`].
pub(super) fn install_interrupt_dispatcher() -> Result<()> {
    use esp_idf_svc::sys::{gpio_install_isr_service, ESP_INTR_FLAG_IRAM, ESP_INTR_FLAG_LEVEL3};

    let iram = ESP_INTR_FLAG_IRAM as i32;
    let level3 = ESP_INTR_FLAG_LEVEL3 as i32;
    let installed = match esp!(unsafe { gpio_install_isr_service(iram | level3) }) {
        Ok(()) => Ok(()),
        Err(error) => {
            log::warn!("DRDY dispatcher: no level-3 slot ({error}); falling back to level 1");
            esp!(unsafe { gpio_install_isr_service(iram) })
        }
    };
    installed?;
    // esp-idf-hal installs the same service lazily from `PinDriver::subscribe`
    // and would fail on the second install; tell it the service exists so the
    // rest of the GPIO code keeps working.
    unsafe { esp_idf_svc::hal::gpio::set_isr_service_flag_unchecked() };
    Ok(())
}

/// The front end's handle on its interrupt-side frame path.
pub(crate) struct FrameReader {
    /// Chip 0's DRDY, which carries the read, and chip 1's, which is counted.
    data_ready_pins: [u8; DEVICE_COUNT],
    frame_device: usize,
    interrupt_enable: u32,
}

impl FrameReader {
    fn hand_back_to_driver(&self) -> Result<()> {
        if OWNER.load(Ordering::Acquire) != OWNER_RAW {
            return Ok(());
        }
        let deadline = unsafe { esp_timer_get_time() } + HANDOFF_TIMEOUT_US;
        while HANDLER_GATE.load(Ordering::Acquire) != 0 {
            if unsafe { esp_timer_get_time() } >= deadline {
                return Err(anyhow!("DRDY handler did not quiesce before SPI handoff"));
            }
            std::hint::spin_loop();
        }
        let reader = unsafe { *READER.0.get() };
        while unsafe { read_register(reader.spi_base, CMD) } & (CMD_UPDATE | CMD_USR) != 0 {
            if unsafe { esp_timer_get_time() } >= deadline {
                return Err(anyhow!(
                    "SPI host did not become idle before driver handoff"
                ));
            }
            std::hint::spin_loop();
        }
        unsafe {
            for chip_select in reader.chip_selects {
                std::ptr::write_volatile(chip_select.set as *mut u32, chip_select.mask);
            }
            write_register(reader.spi_base, DMA_INT_CLR, TRANS_DONE_INT);
            write_register(reader.spi_base, DMA_INT_ENA, self.interrupt_enable);
            write_register(reader.spi_base, DMA_INT_SET, TRANS_DONE_INT);
            spi_device_release_bus(self.frame_device as spi_device_handle_t);
        }
        OWNER.store(OWNER_DRIVER, Ordering::Release);
        Ok(())
    }

    /// Takes ownership of the shared SPI host for interrupt-side reads.
    ///
    /// Both chips must already be streaming (RDATAC, START high) and the SPI
    /// driver must have just performed a frame read on this host through
    /// `frame_device` — that read is what leaves the registers holding the
    /// recipe this snapshots. The interrupt is not enabled here; the caller
    /// validates first ([`Self::validate`]) and then calls [`Self::enable`].
    pub(super) fn claim(
        spi_host: esp_idf_svc::sys::spi_host_device_t,
        frame_device: spi_device_handle_t,
        chip_select_pins: [u8; DEVICE_COUNT],
        data_ready_pins: [u8; DEVICE_COUNT],
    ) -> Result<Self> {
        let spi_base = match spi_host {
            host if host == spi_host_device_t_SPI2_HOST => DR_REG_SPI2_BASE,
            host if host == spi_host_device_t_SPI3_HOST => DR_REG_SPI3_BASE,
            other => return Err(anyhow!("SPI host {other} has no known register base")),
        };
        esp!(unsafe { spi_device_acquire_bus(frame_device, u32::MAX) })?;
        let interrupt_enable = unsafe { read_register(spi_base, DMA_INT_ENA) };
        unsafe {
            write_register(spi_base, DMA_INT_ENA, interrupt_enable & !TRANS_DONE_INT);
            write_register(spi_base, DMA_INT_CLR, TRANS_DONE_INT);
        }
        let reader = HostReader {
            spi_base,
            recipe: Recipe {
                ctrl: unsafe { read_register(spi_base, CTRL) },
                clock: unsafe { read_register(spi_base, CLOCK) },
                user: unsafe { read_register(spi_base, USER) },
                user1: unsafe { read_register(spi_base, USER1) },
                user2: unsafe { read_register(spi_base, USER2) },
                ms_dlen: unsafe { read_register(spi_base, MS_DLEN) },
                misc: unsafe { read_register(spi_base, MISC) },
                din_mode: unsafe { read_register(spi_base, DIN_MODE) },
                din_num: unsafe { read_register(spi_base, DIN_NUM) },
                dout_mode: unsafe { read_register(spi_base, DOUT_MODE) },
                dma_conf: unsafe { read_register(spi_base, DMA_CONF) },
                slave: unsafe { read_register(spi_base, SLAVE) },
                clk_gate: unsafe { read_register(spi_base, CLK_GATE) },
            },
            chip_selects: chip_select_pins.map(ChipSelect::for_pin),
        };
        // The interrupt has never been enabled, so nothing else can be looking.
        unsafe { *READER.0.get() = reader };
        OWNER.store(OWNER_RAW, Ordering::Release);
        Ok(Self {
            data_ready_pins,
            frame_device: frame_device as usize,
            interrupt_enable,
        })
    }

    /// Runs the interrupt's read path once, here in task context, and returns
    /// the frames it produced, one per chip.
    ///
    /// This is the register recipe's proof. The caller compares what comes back
    /// against the frames the SPI driver read moments earlier: same chips, same
    /// stream, so a recipe that clocks the wrong number of bits, picks the wrong
    /// mode, or misses the FIFO shows up as a broken status marker or nonsense
    /// channel codes before the interrupt is ever armed.
    pub(super) fn validate(&self) -> Option<[[u8; FRAME_BYTES]; DEVICE_COUNT]> {
        // Safe by the ownership contract: the interrupt is not enabled yet, so
        // this is the only thing touching the host.
        unsafe { read_frames_into_rings() };
        let first = self.pop(0)?.bytes;
        let second = self.pop(1)?.bytes;
        Some([first, second])
    }

    /// Routes chip 0's DRDY falling edge into the read handler and chip 1's into
    /// the counter. The dispatcher must already be installed
    /// ([`install_interrupt_dispatcher`]); both stay disabled until
    /// [`Self::enable`].
    pub(super) fn attach(&self) -> Result<()> {
        let handlers: [unsafe extern "C" fn(*mut core::ffi::c_void); DEVICE_COUNT] =
            [data_ready_handler, counted_edge_handler];
        for (pin, handler) in self.data_ready_pins.into_iter().zip(handlers) {
            let pin = pin as i32;
            esp!(unsafe { gpio_set_intr_type(pin, gpio_int_type_t_GPIO_INTR_NEGEDGE) })?;
            esp!(unsafe { gpio_intr_disable(pin) })?;
            esp!(unsafe { gpio_isr_handler_add(pin, Some(handler), std::ptr::null_mut()) })?;
        }
        Ok(())
    }

    /// Opens the interrupt window: from here the handler owns the SPI host.
    ///
    /// Unlike esp-idf-hal's subscription, nothing disables the interrupt behind
    /// this call — esp-idf's dispatcher clears the pin's interrupt status before
    /// it runs handlers, so an edge landing during a read is latched and served
    /// next, and there is no re-arm to lose an edge in.
    pub(super) fn enable(&self) -> Result<()> {
        let reader = unsafe { *READER.0.get() };
        if OWNER.load(Ordering::Acquire) == OWNER_DRIVER {
            let frame_device = self.frame_device as spi_device_handle_t;
            esp!(unsafe { spi_device_acquire_bus(frame_device, u32::MAX) })?;
            unsafe {
                write_register(
                    reader.spi_base,
                    DMA_INT_ENA,
                    self.interrupt_enable & !TRANS_DONE_INT,
                );
                write_register(reader.spi_base, DMA_INT_CLR, TRANS_DONE_INT);
            }
            OWNER.store(OWNER_RAW, Ordering::Release);
        }
        HANDLER_GATE.store(HANDLER_GATE_OPEN, Ordering::Release);
        for pin in self.data_ready_pins {
            esp!(unsafe { gpio_intr_enable(pin as i32) })?;
        }
        Ok(())
    }

    /// Closes the interrupt window and waits out any read still in flight, so
    /// the SPI driver can safely take the host back.
    ///
    /// The atomic gate closes admission before the GPIO sources are disabled and
    /// counts every admitted handler, including one executing on the other core.
    pub(super) fn disable(&self) -> Result<()> {
        HANDLER_GATE.fetch_and(!HANDLER_GATE_OPEN, Ordering::AcqRel);
        let mut disabled = Ok(());
        for pin in self.data_ready_pins {
            let result = esp!(unsafe { gpio_intr_disable(pin as i32) });
            if disabled.is_ok() {
                disabled = result;
            }
        }
        self.hand_back_to_driver()?;
        disabled.map_err(Into::into)
    }

    /// The oldest frame the interrupt has captured from chip `index`, or `None`
    /// when its ring is empty. Never blocks.
    pub(super) fn pop(&self, index: usize) -> Option<CapturedFrame> {
        let ring = &RINGS[index];
        let read = ring.read_index.load(Ordering::Relaxed);
        if read == ring.write_index.load(Ordering::Acquire) {
            return None;
        }
        // The producer never writes a slot the consumer can see: it publishes by
        // advancing `write_index` after filling, and it refuses to fill at all
        // once the ring is full.
        let slot = unsafe { (*ring.slots.get())[(read & RING_MASK) as usize] };
        ring.read_index
            .store(read.wrapping_add(1), Ordering::Release);
        Some(CapturedFrame {
            edge_us: slot.edge_us,
            read_us: slot.read_us,
            bytes: words_to_bytes(&slot.words),
        })
    }

    /// Throws away every frame both rings hold. Used across a warm recovery,
    /// whose frames belong to the chips' previous configuration.
    pub(super) fn clear(&self) {
        for ring in &RINGS {
            ring.read_index
                .store(ring.write_index.load(Ordering::Acquire), Ordering::Release);
        }
    }

    /// Cumulative interrupt-side counters for chip `index`: serviced edges, ring
    /// overruns, and transfers that gave up waiting on the host.
    pub(super) fn counters(&self, index: usize) -> InterruptCounters {
        let ring = &RINGS[index];
        InterruptCounters {
            edges: ring.edges.load(Ordering::Relaxed),
            overruns: ring.overruns.load(Ordering::Relaxed),
            read_faults: ring.read_faults.load(Ordering::Relaxed),
        }
    }

    /// Chip 1's DRDY falling edges, cumulative.
    pub(super) fn counted_edges(&self) -> u32 {
        COUNTED_EDGES.load(Ordering::Relaxed)
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        HANDLER_GATE.fetch_and(!HANDLER_GATE_OPEN, Ordering::AcqRel);
        for pin in self.data_ready_pins {
            let _ = unsafe { gpio_intr_disable(pin as i32) };
        }
        if self.hand_back_to_driver().is_err() {
            std::process::abort();
        }
    }
}

/// One frame as it left the interrupt.
pub(super) struct CapturedFrame {
    /// Device clock read at the top of the handler — the DRDY edge to within
    /// interrupt entry latency, which is the truest timestamp this firmware can
    /// produce for a conversion.
    pub(super) edge_us: u64,
    /// Handler time from that stamp to the end of the transfer.
    pub(super) read_us: u32,
    pub(super) bytes: [u8; FRAME_BYTES],
}

/// What the interrupt side has counted since boot.
pub(super) struct InterruptCounters {
    pub(super) edges: u32,
    pub(super) overruns: u32,
    pub(super) read_faults: u32,
}

/// FIFO words to wire bytes. The GPSPI data registers hold received bytes
/// little-endian within each word, first byte in the low octet, so the frame is
/// the concatenation of `to_le_bytes` truncated to the 27 the chip sent.
fn words_to_bytes(words: &[u32; FRAME_WORDS]) -> [u8; FRAME_BYTES] {
    let mut bytes = [0u8; FRAME_BYTES];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = (words[index / 4] >> (8 * (index % 4))) as u8;
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_capacity_is_a_power_of_two() {
        assert!(RING_CAPACITY.is_power_of_two());
        assert_eq!(RING_MASK as usize, RING_CAPACITY - 1);
    }

    /// The frame is 27 bytes and the FIFO deals in words: seven words carry it,
    /// and the last byte of the last word is not part of the frame.
    #[test]
    fn a_frame_occupies_seven_fifo_words() {
        assert_eq!(FRAME_WORDS, 7);
        assert_eq!(FRAME_BYTES, 27);
    }

    /// Byte order across the word boundary is the part that would silently
    /// scramble every channel, so it is pinned by number.
    #[test]
    fn fifo_words_unpack_first_byte_from_the_low_octet() {
        let mut words = [0u32; FRAME_WORDS];
        words[0] = 0x4433_22C0;
        words[6] = 0x0000_0077;
        let bytes = words_to_bytes(&words);
        assert_eq!(bytes[0], 0xC0);
        assert_eq!(bytes[1], 0x22);
        assert_eq!(bytes[2], 0x33);
        assert_eq!(bytes[3], 0x44);
        assert_eq!(bytes[24], 0x77);
        assert_eq!(bytes[26], 0x00);
    }
}
