//! The process data exchange, abstracted so that the bus thread, the controllers and the exit
//! sequence run the same against the real robot (ethercrab, src/lifecycle.rs) and the simulated
//! one (src/sim.rs).

use crate::pdo::{IN_LEN, OUT_LEN};

#[derive(Clone, Debug)]
pub enum BusError {
    /// No answer within the PDU timeout (frame lost, or the robot has left EtherCAT).
    Timeout,
    /// The robot is not on the bus any more (simulator after DisableEtherCAT).
    Gone,
    Other(String),
}

impl std::fmt::Display for BusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BusError::Timeout => write!(f, "timeout"),
            BusError::Gone => write!(f, "subdevice gone"),
            BusError::Other(s) => write!(f, "{s}"),
        }
    }
}

pub trait Fieldbus: Send {
    /// One cycle: send the 60 output bytes, receive the 252 input bytes. Returns the working
    /// counter (0: the robot did not process the frame; the inputs are then not valid).
    fn exchange(&mut self, out: &[u8; OUT_LEN], inp: &mut [u8; IN_LEN]) -> Result<u16, BusError>;

    fn is_sim(&self) -> bool {
        false
    }
}

impl<T: Fieldbus + ?Sized> Fieldbus for Box<T> {
    fn exchange(&mut self, out: &[u8; OUT_LEN], inp: &mut [u8; IN_LEN]) -> Result<u16, BusError> {
        (**self).exchange(out, inp)
    }

    fn is_sim(&self) -> bool {
        (**self).is_sim()
    }
}

/// A fieldbus made of a closure (the ethercrab glue builds one without naming its group type).
pub struct FnBus<F>(pub F);

impl<F> Fieldbus for FnBus<F>
where
    F: FnMut(&[u8; OUT_LEN], &mut [u8; IN_LEN]) -> Result<u16, BusError> + Send,
{
    fn exchange(&mut self, out: &[u8; OUT_LEN], inp: &mut [u8; IN_LEN]) -> Result<u16, BusError> {
        (self.0)(out, inp)
    }
}
