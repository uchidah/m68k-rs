use m68k::core::memory::{AddressBus, LinearMemoryBus};
use m68k::{CpuCore, CpuType, CycleBatchExit};

struct BoundaryBus {
    inner: LinearMemoryBus,
    requested: bool,
}

impl BoundaryBus {
    fn new(requested: bool) -> Self {
        Self {
            inner: LinearMemoryBus::new(0x1000),
            requested,
        }
    }

    fn load(&mut self, address: u32, bytes: &[u8]) {
        self.inner.load(address, bytes);
    }
}

impl AddressBus for BoundaryBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.inner.read_byte(address)
    }

    fn read_word(&mut self, address: u32) -> u16 {
        self.inner.read_word(address)
    }

    fn read_long(&mut self, address: u32) -> u32 {
        self.inner.read_long(address)
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.inner.write_byte(address, value);
    }

    fn write_word(&mut self, address: u32, value: u16) {
        self.inner.write_word(address, value);
    }

    fn write_long(&mut self, address: u32, value: u32) {
        self.inner.write_long(address, value);
    }

    fn instruction_boundary_requested(&mut self) -> bool {
        std::mem::take(&mut self.requested)
    }
}

fn cpu_at(address: u32) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.pc = address;
    cpu
}

#[test]
fn bus_boundary_stops_after_decoded_instruction_with_cycles() {
    let mut cpu = cpu_at(0x100);
    let mut bus = BoundaryBus::new(true);
    bus.load(0x100, &[0x4E, 0x71, 0x4E, 0x71]);

    let result = cpu.run_until_cycles(&mut bus, 100, &[]);

    assert_eq!(result.instructions, 1);
    assert_eq!(result.cycles, 4);
    assert_eq!(result.exit, CycleBatchExit::BusRequestedBoundary);
    assert_eq!(cpu.pc, 0x102);
}

#[test]
fn bus_boundary_stops_after_full_dispatch_with_cycles() {
    let mut cpu = cpu_at(0x100);
    cpu.set_a(0, 0x200);
    let mut bus = BoundaryBus::new(true);
    bus.load(0x100, &[0x30, 0x10, 0x4E, 0x71]); // MOVE.W (A0),D0
    bus.load(0x200, &[0x12, 0x34]);

    let result = cpu.run_until_cycles(&mut bus, 100, &[]);

    assert_eq!(result.instructions, 1);
    assert_eq!(result.cycles, 8);
    assert_eq!(result.exit, CycleBatchExit::BusRequestedBoundary);
    assert_eq!(cpu.d(0), 0x1234);
}
