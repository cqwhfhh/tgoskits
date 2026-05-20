use core::mem::{align_of, size_of};

use ax_errno::{AxError, AxResult};
use ax_task::current;
use starry_vm::VmMutPtr;

use crate::task::{AsThread, RseqRegistration};

const RSEQ_FLAG_UNREGISTER: u32 = 1;
const RSEQ_MIN_LEN: usize = size_of::<Rseq>();
const RSEQ_ALIGNMENT: usize = align_of::<Rseq>();
const RSEQ_UNINITIALIZED_CPU_ID: u32 = u32::MAX;

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
struct Rseq {
    cpu_id_start: u32,
    cpu_id: u32,
    rseq_cs: u64,
    flags: u32,
    node_id: u32,
    mm_cid: u32,
}

impl Rseq {
    const fn registered() -> Self {
        Self {
            cpu_id_start: RSEQ_UNINITIALIZED_CPU_ID,
            cpu_id: RSEQ_UNINITIALIZED_CPU_ID,
            rseq_cs: 0,
            flags: 0,
            node_id: 0,
            mm_cid: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RseqOp {
    Register(RseqRegistration),
    Unregister { sig: u32 },
}

fn validate_rseq_area(addr: *mut u8, len: usize) -> AxResult<usize> {
    if addr.is_null() {
        return Err(AxError::InvalidInput);
    }
    if len < RSEQ_MIN_LEN || len > u32::MAX as usize {
        return Err(AxError::InvalidInput);
    }

    let addr = addr.addr();
    if !addr.is_multiple_of(RSEQ_ALIGNMENT) {
        return Err(AxError::InvalidInput);
    }

    Ok(addr)
}

fn parse_rseq_request(addr: *mut u8, len: usize, flags: u32, sig: u32) -> AxResult<RseqOp> {
    match flags {
        0 => Ok(RseqOp::Register(RseqRegistration {
            addr: validate_rseq_area(addr, len)?,
            len,
            sig,
        })),
        RSEQ_FLAG_UNREGISTER => {
            if !addr.is_null() || len != 0 {
                return Err(AxError::InvalidInput);
            }
            Ok(RseqOp::Unregister { sig })
        }
        _ => Err(AxError::InvalidInput),
    }
}

fn resolve_rseq_transition(
    current: Option<RseqRegistration>,
    request: RseqOp,
) -> AxResult<Option<RseqRegistration>> {
    match request {
        RseqOp::Register(next) => {
            if let Some(current) = current {
                if current.addr != next.addr || current.len != next.len {
                    return Err(AxError::InvalidInput);
                }
                if current.sig != next.sig {
                    return Err(AxError::OperationNotPermitted);
                }
                return Err(AxError::ResourceBusy);
            }
            Ok(Some(next))
        }
        RseqOp::Unregister { sig } => {
            let Some(current) = current else {
                return Err(AxError::InvalidInput);
            };
            if current.sig != sig {
                return Err(AxError::OperationNotPermitted);
            }
            Ok(None)
        }
    }
}

/// Minimal `rseq(2)` handling.
///
/// The kernel now accepts the legacy registration/unregistration path and
/// initializes the userspace control block, but it still does not provide the
/// full restart-abort machinery that Linux uses for CPU migration handling.
///
/// C prototype (simplified):
/// long rseq(void *addr, uint32_t len, int flags, uint32_t sig);
pub fn sys_rseq(addr: *mut u8, len: usize, flags: u32, sig: u32) -> AxResult<isize> {
    debug!(
        "sys_rseq <= addr: {:?}, len: {}, flags: {}, sig: {}",
        addr, len, flags, sig
    );

    let request = parse_rseq_request(addr, len, flags, sig)?;
    let thread = current().as_thread();
    let next = resolve_rseq_transition(thread.rseq_registration(), request)?;

    if let Some(registration) = next {
        (registration.addr as *mut Rseq).vm_write(Rseq::registered())?;
        thread.set_rseq_registration(registration.addr, registration.len, registration.sig);
    } else {
        thread.clear_rseq_registration();
    }

    Ok(0)
}

#[cfg(test)]
mod tests {
    use ax_errno::AxError;

    use super::{
        RSEQ_FLAG_UNREGISTER, RseqRegistration, parse_rseq_request, resolve_rseq_transition,
        validate_rseq_area,
    };

    #[test]
    fn validate_rseq_area_accepts_32_byte_aligned_region() {
        let mut buf = [0u8; 64];
        let ptr = buf.as_mut_ptr();
        assert_eq!(validate_rseq_area(ptr, 32).unwrap(), ptr.addr());
    }

    #[test]
    fn validate_rseq_area_rejects_short_region() {
        let mut buf = [0u8; 64];
        assert_eq!(
            validate_rseq_area(buf.as_mut_ptr(), 16).unwrap_err(),
            AxError::InvalidInput
        );
    }

    #[test]
    fn validate_rseq_area_rejects_null_pointer() {
        assert_eq!(
            validate_rseq_area(core::ptr::null_mut(), 32).unwrap_err(),
            AxError::InvalidInput
        );
    }

    #[test]
    fn validate_rseq_area_rejects_misaligned_region() {
        let mut buf = [0u8; 64];
        let ptr = buf.as_mut_ptr().wrapping_add(1);
        assert_eq!(
            validate_rseq_area(ptr, 32).unwrap_err(),
            AxError::InvalidInput
        );
    }

    #[test]
    fn parse_rseq_request_handles_unregister() {
        match parse_rseq_request(core::ptr::null_mut(), 0, RSEQ_FLAG_UNREGISTER, 7).unwrap() {
            super::RseqOp::Unregister { sig } => assert_eq!(sig, 7),
            _ => panic!("expected unregister request"),
        }
    }

    #[test]
    fn resolve_rseq_transition_accepts_initial_register() {
        let request = RseqRegistration {
            addr: 0x1000,
            len: 32,
            sig: 9,
        };
        assert_eq!(
            resolve_rseq_transition(None, super::RseqOp::Register(request)).unwrap(),
            Some(request)
        );
    }

    #[test]
    fn resolve_rseq_transition_rejects_duplicate_register() {
        let request = RseqRegistration {
            addr: 0x1000,
            len: 32,
            sig: 9,
        };
        assert_eq!(
            resolve_rseq_transition(Some(request), super::RseqOp::Register(request)).unwrap_err(),
            AxError::ResourceBusy
        );
    }

    #[test]
    fn resolve_rseq_transition_rejects_sig_mismatch() {
        let current = RseqRegistration {
            addr: 0x1000,
            len: 32,
            sig: 9,
        };
        let request = RseqRegistration {
            addr: 0x1000,
            len: 32,
            sig: 10,
        };
        assert_eq!(
            resolve_rseq_transition(Some(current), super::RseqOp::Register(request)).unwrap_err(),
            AxError::OperationNotPermitted
        );
    }

    #[test]
    fn resolve_rseq_transition_rejects_unregister_sig_mismatch() {
        let current = RseqRegistration {
            addr: 0x1000,
            len: 32,
            sig: 9,
        };
        assert_eq!(
            resolve_rseq_transition(Some(current), super::RseqOp::Unregister { sig: 8 })
                .unwrap_err(),
            AxError::OperationNotPermitted
        );
    }
}
