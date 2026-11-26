//! meca_ecat: the latency tests of meca_latency_test.py over EtherCAT (Linux RT, ethercrab).
//! See ecat/README.md. The Python side (ecat_bridge.py) imports and analyses the runs.

mod log;
mod bus;
mod cli;
mod controllers;
mod fieldbus;
mod lifecycle;
mod out;
mod pdo;
mod robot;
mod rt;
mod sim;
mod supervisor;
mod tcp;

#[cfg(feature = "suite")]
mod actionref;
#[cfg(feature = "suite")]
mod bridge;
#[cfg(feature = "suite")]
mod stream;
#[cfg(feature = "suite")]
mod suite;
#[cfg(feature = "suite")]
mod worker;

use std::process::ExitCode;

use clap::Parser;

use cli::{Cli, Command};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.cmd {
        #[cfg(feature = "suite")]
        Command::Run(a) => supervisor::run(a),
        #[cfg(feature = "suite")]
        Command::Worker(a) => worker::run(a),
        Command::Probe(p) => supervisor::probe(p),
        Command::Restore(r) => supervisor::restore_cmd(r),
        Command::Status(o) => supervisor::status_cmd(o),
        Command::Burn { secs } => supervisor::burn(secs),
    }
}
