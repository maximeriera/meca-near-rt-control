# meca_ecat: the latency tests over EtherCAT

`meca_ecat` runs the tests of `meca_latency_test.py` on a Meca500 over EtherCAT, from a Linux
PREEMPT_RT station, with a 1 ms cycle. It only drives the robot and records. The run folders it
writes are imported by `ecat_bridge.py` into the v2 format of `meca_latency_test.py`. The analysis,
the plots, `summary.json` and `aggregate.py` are therefore the same as for the TCP runs, and the two
can be compared directly. `aggregate.py` adds a `transport` column to tell them apart.

**Tests ported:** step, sine, accsweep, multijoint, servo, servomulti, replay (vel, ff, pos, posi),
soak, and timing, which is always on. Options keep the names and defaults of `meca_latency_test.py`,
with two exceptions:
- `--period` defaults to 1 ms, one cycle;
- `--joint-acc-pct` is always applied (150 by default, the value of the TCP campaign; the robot
  takes 100 at activation), because it cannot be read back over EtherCAT.

## The way back to TCP/IP (MecaPortal)

EtherCAT mode is persistent on the robot: a power cycle does not bring TCP/IP back. Every run
therefore ends the same way, on every path. That includes normal end, test error, safety stop,
Ctrl-C, SIGTERM, SIGHUP (SSH dropped), a panic in the test code, and the bus thread dying.

1. Stop: velocity zero, queued motion cleared.
2. Deactivate.
3. `DisableEtherCAT` (0x7200:07, output byte 0 bit 6), held until the robot leaves the bus. Until
   the byte order is confirmed, the bit is also written where the other order would put it.
4. Wait until TCP port 10000 answers (`--tcp-wait`, 120 s). This is the only proof of success: a
   dead bus looks the same as a robot that has left EtherCAT.
5. If port 10000 does not answer: `restore` in the same process. It starts a fresh master, goes from
   INIT to OP, then Deactivate and DisableEtherCAT. If OP is refused, it writes DisableEtherCAT over
   CoE (SDO) instead.

`run` is a supervisor. It starts the test process and, if that process dies without confirming the
robot is back on TCP/IP (kill -9, abort, OOM), runs `restore` itself.

**Signals during a run:**
- 1st Ctrl-C: the current test stops cleanly, then the exit sequence runs.
- 2nd Ctrl-C: the bus thread starts the exit sequence itself, right away.
- The exit sequence is never skipped. Only Ctrl-\ kills the test process, and the supervisor then
  restores.

**If the robot is still not on TCP/IP:**
```
meca_ecat restore --iface <IF>          # fresh master, OP, Deactivate, DisableEtherCAT
meca_ecat restore --iface <IF> --sdo    # DisableEtherCAT written over CoE in PRE-OP
```
As a last resort, do the network configuration reset (Meca500 user manual, troubleshooting):
1. Power off.
2. On the R4, engage the E-stop.
3. Hold the base Power button while powering on, about 40 s, until the red and yellow LEDs blink in
   sequence.

The robot comes back on 192.168.0.100 in TCP/IP mode.

## Station setup (once)

- **Robot:** firmware 11.3.3 (it fixes a high latency of velocity commands at 1 kHz over EtherCAT).
  The cable goes on **ETHERNET1** (EtherCAT IN), alone on a dedicated port of the PC, with no
  switch.
- **Network:**
  - The dedicated port needs a static address in the robot's subnet (e.g. 192.168.0.10/24). Before
    and after the EtherCAT session, the robot is reached over TCP/IP through that same port, and
    `meca_ecat` checks the route.
  - Turn interrupt moderation and EEE off: `sudo ethtool -C <IF> rx-usecs 0`, `sudo ethtool --set-eee <IF> eee off`.
  - Optionally, isolate two cores (`isolcpus=2,3 nohz_full=2,3`) and move the NIC interrupts off them.
- **Rust:** rustup, stable toolchain >= 1.85.
- **Python:** python3 with numpy, scipy and matplotlib for the analysis. Also mecademicpy, to
  re-analyse the TCP runs (their pickles hold a mecademicpy enum).

## Build

The crate has never been compiled yet: it was written on the Windows PC.

```
cd ecat
cargo build --release --no-default-features   # life cycle tools only: probe, restore, status
cargo test --no-default-features
cargo build --release && cargo test            # + the test suite
./setcap.sh                                    # after EVERY build (capabilities are reset)
```

## First steps on the robot (in this order)

1. **No motion:** run `probe` three times.
   ```
   target/release/meca_ecat probe --iface <IF>
   ```
   It switches to EtherCAT, detects the byte order, shows the decoded feedback for 5 s, then comes
   back to TCP/IP. Check that the joint values match the MecaPortal and that the MecaPortal opens
   afterwards. Then test `restore --sdo`.
2. **Failure cases, no motion.** Each one must end with port 10000 answering, by the run itself, by
   the supervisor, or by `restore`:
   - SIGSTOP of the worker for 1 s (SM watchdog);
   - cable pulled and plugged back;
   - E-stop pressed;
   - robot in error;
   - SSH session dropped;
   - `kill -9` of the worker;
   - `kill -9` during the DisableEtherCAT hold.
3. **Without arm motion:** `--robot-sim` activates the robot in its own simulation mode, so the real
   EtherCAT link is used but the arm does not move.
   ```
   meca_ecat run --iface <IF> --robot-sim --tests step,servo
   ```
4. **Low speed, E-stop at hand:**
   ```
   meca_ecat run --iface <IF> --tests step --vel-limit-pct 10 --repeats 2
   ```
   Then check bus loss: the bus thread stops for 200 ms (SIGSTOP) and the robot must stop. This
   behaviour is undocumented.
5. **Campaign:**
   ```
   meca_ecat run --iface <IF> --tests all --bus-cpu 2 --txrx-cpu 3
   meca_ecat run --iface <IF> --tests soak --bus-cpu 2 --txrx-cpu 3
   python3 aggregate.py --runs "meca_latency_*"   # TCP 11.3.3 vs EtherCAT 11.3.3
   ```

## Dry run without a robot

```
meca_ecat run --sim --tests step,sine,servo,servomulti,replay,soak --soak-dur 5
```
The simulated robot is crude: it follows the J6 law of SimIO, has a free-running clock drifting
40 ppm, and implements the cyclic protocol. `--sim-byte-order be` exercises the byte order
detection.

## How it works

| Thread | Role |
|---|---|
| tx/rx (ethercrab `tx_rx_task`, SCHED_FIFO 90, `--txrx-cpu`) | the raw socket |
| bus (SCHED_FIFO 80, `--bus-cpu`, `clock_nanosleep` 1 ms) | at each cycle, the active controller computes the outputs from the newest feedback, then `tx_rx`, decoding, recording, safety guard |
| writer | stores frames and bus cycles (lock-free rings from the bus) |
| orchestrator | the tests in sequence (`suite.rs`, a port of `Tester`), the calls to `ecat_bridge.py` between tests, the exit sequence |

- **Logs:** a log row holds `ts` (cycle deadline), `t1` and `t2` (around the exchange that carries
  the command) and the robot / host time of the frame it was computed from. A frame's host time is
  the `t2` of the cycle that brought the new robot sample.
- **Safety guard:** checked at every cycle on test motions. It trips on max excursion, robot error
  or E-stop. Then velocity goes to zero, queued motion is cleared, the test stops, and the exit
  sequence runs.
- **Clock map** (`meca_latency_test.clock_map_ecat`, used when `raw._ecat` is present): the robot
  and the master each tick at 1 ms, so the data age is a sawtooth that wraps every 1/drift. The map
  is the edge of the lower convex hull of host_rx - robot_ts. `ecat_timing` gives the drift and the
  number of beats covered: below one beat, robot-clock delays may read up to one cycle late.
- **Replay and soak references** come from `ecat_bridge.py` (the same numpy RNG as the TCP runs).
  The quick model after the step test (feedforward advance) also comes from there.

## Still to confirm on the robot

- the byte order of the cyclic data (the manual says big-endian, the ESI is standard);
- what DisableEtherCAT does exactly (reboot or not, how long, which states accept it);
- acknowledgement of AL errors by ethercrab 0.7.1 (re-init from INIT is used);
- the robot's reaction to the loss of the master in velocity mode;
- the effect of DC SYNC0 (not used: SM-synchronous);
- the real MoveJoints rate with one command in flight.
