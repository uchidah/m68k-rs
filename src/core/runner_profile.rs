//! Deterministic, low-intervention sampling for boundary-hook execution.
//!
//! This module is enabled only by the `runner-profile` feature. It records no
//! timestamps for instructions that do not match the configured interval.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::types::CpuType;

/// The narrow dispatcher path selected for a sampled instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchKind {
    /// The normal dispatcher handled the already-fetched opcode.
    Fallback,
    /// The boundary dispatcher handled NOP.
    Nop,
    /// The boundary dispatcher handled MOVEQ.
    Moveq,
    /// The boundary dispatcher handled CLR.W Dn.
    ClrWord,
    /// The boundary dispatcher handled CLR.L Dn.
    ClrLong,
    /// The boundary dispatcher handled CMP.W #imm,Dn on M68000.
    CmpWordImmediate,
}

/// Bus-operation counts recorded only while a timing sample is active.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BusCounts {
    /// Byte reads during the sampled instruction.
    pub read_byte: u32,
    /// Word reads during the sampled instruction.
    pub read_word: u32,
    /// Long reads during the sampled instruction.
    pub read_long: u32,
    /// Byte writes during the sampled instruction.
    pub write_byte: u32,
    /// Word writes during the sampled instruction.
    pub write_word: u32,
    /// Long writes during the sampled instruction.
    pub write_long: u32,
    /// Reads observed before opcode fetch completed.
    pub fetch_reads: u32,
    /// Reads and writes observed after opcode fetch completed.
    pub data_bus_ops: u32,
}

/// One completed sampled instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionSample {
    /// One-based deterministic sample index.
    pub index: u64,
    /// Program counter of the completed instruction.
    pub ppc: u32,
    /// Fetched opcode.
    pub opcode: u16,
    /// Numeric CPU model discriminator.
    pub cpu_type: u8,
    /// Dispatcher classification.
    pub dispatch: DispatchKind,
    /// Instruction cycles after finalization.
    pub cycles: i32,
    /// Time from instruction entry through precise opcode fetch.
    pub fetch_prepare_ns: u64,
    /// Time from post-fetch dispatch through instruction-specific execution.
    pub dispatch_body_ns: u64,
    /// Time spent in precise finalization after dispatch.
    pub finalize_boundary_ns: u64,
    /// Cheap bus-operation counters for the sampled instruction.
    pub bus: BusCounts,
}

/// Aggregate data returned after a profiling run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunnerProfileSnapshot {
    /// Deterministic interval in retired-instruction attempts.
    pub interval: u32,
    /// Number of calls that reached the interval decision.
    pub observed_instructions: u64,
    /// Number of completed normal instruction samples.
    pub instruction_samples: u64,
    /// Number of interrupt entries timed by the low-frequency observer.
    pub interrupt_samples: u64,
    /// Total interrupt-entry service time.
    pub interrupt_nanoseconds: u64,
    /// Completed samples, in execution order.
    pub samples: Vec<InstructionSample>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BusPhase {
    Fetch,
    Dispatch,
    Finalize,
}

struct ActiveSample {
    index: u64,
    started: Instant,
    fetch_done: Option<Instant>,
    dispatch_done: Option<Instant>,
    ppc: u32,
    opcode: u16,
    cpu_type: u8,
    dispatch: DispatchKind,
    phase: BusPhase,
    next_bus_read_is_fetch: bool,
    bus: BusCounts,
}

struct RunnerProfile {
    instruction_samples: u64,
    interrupt_samples: u64,
    interrupt_nanoseconds: u64,
    samples: Vec<InstructionSample>,
    active: Option<ActiveSample>,
}

impl Default for RunnerProfile {
    fn default() -> Self {
        Self {
            instruction_samples: 0,
            interrupt_samples: 0,
            interrupt_nanoseconds: 0,
            samples: Vec::new(),
            active: None,
        }
    }
}

thread_local! {
    static PROFILE: RefCell<RunnerProfile> = RefCell::new(RunnerProfile::default());
    static INTERVAL: Cell<u32> = const { Cell::new(0) };
    static UNTIL_SAMPLE: Cell<u32> = const { Cell::new(0) };
    static OBSERVED_INSTRUCTIONS: Cell<u64> = const { Cell::new(0) };
}

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Resets sampling state and reserves storage before execution starts.
pub fn reset(interval: u32, capacity: usize) {
    ENABLED.store(false, Ordering::Release);
    PROFILE.with_borrow_mut(|profile| {
        *profile = RunnerProfile {
            samples: Vec::with_capacity(capacity),
            ..RunnerProfile::default()
        };
    });
    INTERVAL.with(|value| value.set(interval));
    UNTIL_SAMPLE.with(|value| value.set(interval));
    OBSERVED_INSTRUCTIONS.with(|value| value.set(0));
    ENABLED.store(interval != 0, Ordering::Release);
}

/// Returns all aggregate and raw sample data accumulated so far.
pub fn snapshot() -> RunnerProfileSnapshot {
    PROFILE.with_borrow(|profile| RunnerProfileSnapshot {
        interval: INTERVAL.with(Cell::get),
        observed_instructions: OBSERVED_INSTRUCTIONS.with(Cell::get),
        instruction_samples: profile.instruction_samples,
        interrupt_samples: profile.interrupt_samples,
        interrupt_nanoseconds: profile.interrupt_nanoseconds,
        samples: profile.samples.clone(),
    })
}

/// Makes the next instruction sample decision without reading the clock unless selected.
#[inline]
pub fn begin_instruction() {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let selected = UNTIL_SAMPLE.with(|until_sample| {
        OBSERVED_INSTRUCTIONS.with(|observed| observed.set(observed.get().saturating_add(1)));
        if until_sample.get() > 1 {
            until_sample.set(until_sample.get() - 1);
            false
        } else {
            until_sample.set(INTERVAL.with(Cell::get));
            true
        }
    });
    if !selected {
        return;
    }
    PROFILE.with_borrow_mut(|profile| {
        profile.active = Some(ActiveSample {
            index: profile.instruction_samples + 1,
            started: Instant::now(),
            fetch_done: None,
            dispatch_done: None,
            ppc: 0,
            opcode: 0,
            cpu_type: 0,
            dispatch: DispatchKind::Fallback,
            phase: BusPhase::Fetch,
            next_bus_read_is_fetch: false,
            bus: BusCounts::default(),
        });
    });
}

/// Marks the end of the shared precise opcode-fetch prologue.
#[inline]
pub fn finish_fetch(ppc: u32, opcode: u16, cpu_type: CpuType) {
    PROFILE.with_borrow_mut(|profile| {
        if let Some(active) = profile.active.as_mut() {
            active.ppc = ppc;
            active.opcode = opcode;
            active.cpu_type = cpu_type as u8;
            active.fetch_done = Some(Instant::now());
            active.phase = BusPhase::Dispatch;
        }
    });
}

/// Records the dispatcher kind selected after the shared opcode fetch.
#[inline]
pub fn set_dispatch_kind(dispatch: DispatchKind) {
    PROFILE.with_borrow_mut(|profile| {
        if let Some(active) = profile.active.as_mut() {
            active.dispatch = dispatch;
        }
    });
}

/// Marks the end of dispatch and instruction-specific execution.
#[inline]
pub fn finish_dispatch_body() {
    PROFILE.with_borrow_mut(|profile| {
        if let Some(active) = profile.active.as_mut() {
            active.dispatch_done = Some(Instant::now());
            active.phase = BusPhase::Finalize;
        }
    });
}

/// Completes an instruction sample after precise finalization.
#[inline]
pub fn finish_instruction(cycles: i32) {
    PROFILE.with_borrow_mut(|profile| {
        let Some(active) = profile.active.take() else {
            return;
        };
        let now = Instant::now();
        let fetch_done = active.fetch_done.unwrap_or(active.started);
        let dispatch_done = active.dispatch_done.unwrap_or(fetch_done);
        profile.samples.push(InstructionSample {
            index: active.index,
            ppc: active.ppc,
            opcode: active.opcode,
            cpu_type: active.cpu_type,
            dispatch: active.dispatch,
            cycles,
            fetch_prepare_ns: elapsed_ns(active.started, fetch_done),
            dispatch_body_ns: elapsed_ns(fetch_done, dispatch_done),
            finalize_boundary_ns: elapsed_ns(dispatch_done, now),
            bus: active.bus,
        });
        profile.instruction_samples += 1;
    });
}

/// Drops a sample that did not complete as a normal instruction.
#[inline]
pub fn discard_instruction() {
    PROFILE.with_borrow_mut(|profile| profile.active = None);
}

/// Counts a bus operation for the active sample without performing timing.
#[inline]
pub fn note_bus_read(size: u8) {
    PROFILE.with_borrow_mut(|profile| {
        let Some(active) = profile.active.as_mut() else {
            return;
        };
        match size {
            1 => active.bus.read_byte += 1,
            2 => active.bus.read_word += 1,
            4 => active.bus.read_long += 1,
            _ => return,
        }
        if active.next_bus_read_is_fetch {
            active.next_bus_read_is_fetch = false;
            active.bus.fetch_reads += 1;
        } else if active.phase == BusPhase::Dispatch {
            active.bus.data_bus_ops += 1;
        } else {
            active.bus.fetch_reads += 1;
        }
    });
}

/// Marks the next host bus read as a 68000 prefetch-queue refill.
///
/// The precise prefetch engine deliberately uses the regular word-read bus
/// operation so hosts retain their normal bus semantics. This marker lets the
/// optional observer classify that one read without changing the bus API.
#[inline]
pub fn mark_next_bus_read_as_fetch() {
    PROFILE.with_borrow_mut(|profile| {
        if let Some(active) = profile.active.as_mut() {
            active.next_bus_read_is_fetch = true;
        }
    });
}

/// Counts an instruction-stream bus read for the active sample.
#[inline]
pub fn note_fetch_read(size: u8) {
    PROFILE.with_borrow_mut(|profile| {
        let Some(active) = profile.active.as_mut() else {
            return;
        };
        match size {
            1 => active.bus.read_byte += 1,
            2 => active.bus.read_word += 1,
            4 => active.bus.read_long += 1,
            _ => return,
        }
        active.bus.fetch_reads += 1;
    });
}

/// Counts a bus write for the active sample without performing timing.
#[inline]
pub fn note_bus_write(size: u8) {
    PROFILE.with_borrow_mut(|profile| {
        let Some(active) = profile.active.as_mut() else {
            return;
        };
        match size {
            1 => active.bus.write_byte += 1,
            2 => active.bus.write_word += 1,
            4 => active.bus.write_long += 1,
            _ => return,
        }
        active.bus.data_bus_ops += 1;
    });
}

/// Returns an interrupt-entry start timestamp only when an IRQ is pending.
#[inline]
pub fn begin_interrupt(level: u32) -> Option<Instant> {
    (level != 0).then(Instant::now)
}

/// Adds a completed interrupt-entry duration when the CPU accepted it.
#[inline]
pub fn finish_interrupt(start: Option<Instant>, serviced: bool) {
    if !serviced {
        return;
    }
    let Some(start) = start else {
        return;
    };
    PROFILE.with_borrow_mut(|profile| {
        profile.interrupt_samples += 1;
        profile.interrupt_nanoseconds = profile
            .interrupt_nanoseconds
            .saturating_add(elapsed_ns(start, Instant::now()));
    });
}

#[inline]
fn elapsed_ns(start: Instant, end: Instant) -> u64 {
    end.duration_since(start)
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}
