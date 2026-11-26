//! Meca500 EtherCAT process data (ESI v11.3.1 = v11.3.3, copy in ecat/esi/): offsets, bits,
//! command ids, encoding / decoding, byte order detection, and the 54-column frame row of
//! meca_latency_test.py. All PDOs are fixed: 60 output bytes (SM2), 252 input bytes (SM3).

pub const OUT_LEN: usize = 60;
pub const IN_LEN: usize = 252;
pub const VENDOR_ID: u32 = 0x00EC_ADE1;
pub const PRODUCT_CODE: u32 = 0x0004_D500;
pub const NJ: usize = 6;

/// Byte offsets in the process images (checked against the ESI by a unit test).
pub mod off {
    // outputs (RxPDO, master -> robot)
    pub const ROBOT_CONTROL: usize = 0; // 0x7200:1..7, 7 BOOL + 25 padding bits
    pub const MOVE_ID: usize = 4; // 0x7310:1 UINT
    pub const MOTION_CONTROL: usize = 6; // 0x7310:2..5, 4 BOOL + 12 padding bits
    pub const MOVE_COMMAND: usize = 8; // 0x7305:0 UDINT
    pub const MOVE_ARGS: usize = 12; // 0x7306:1..6 REAL
    pub const HOST_TIME: usize = 36; // 0x7400:0 UDINT
    pub const BRAKES_CONTROL: usize = 40; // 0x7410:1..2, 2 BOOL + 30 padding bits
    pub const DYN_CFG: usize = 44; // 0x7420..0x7423:1 UDINT
    // inputs (TxPDO, robot -> master)
    pub const ROBOT_STATUS: usize = 0; // 0x6010:2..10, 9 BOOL + 7 padding bits
    pub const ERROR_CODE: usize = 2; // 0x6010:1 UINT
    pub const CHECKPOINT_REACHED: usize = 4; // 0x6015:1
    pub const CHECKPOINT_DISCARDED: usize = 6; // 0x6015:2
    pub const MOVE_ID_ECHO: usize = 8; // 0x6015:3
    pub const FIFO_SPACE: usize = 10; // 0x6015:4
    pub const MOTION_STATUS: usize = 12; // 0x6015:6..11, 6 BOOL + 10 padding bits
    pub const OFFLINE_PROGRAM_ID: usize = 14; // 0x6015:5
    pub const JOINT_SET: usize = 16; // 0x6030:1..6 REAL, target joint positions [deg]
    pub const POSE: usize = 40; // 0x6031:1..6 REAL, target end-effector pose
    pub const TIMESTAMP_S: usize = 116; // 0x6060:1 UDINT
    pub const TIMESTAMP_US: usize = 120; // 0x6060:2 UDINT
    pub const DYN_CYCLES: usize = 124; // 0x6060:3 UDINT
    pub const SAFETY: usize = 128; // 0x6065, 12 bytes
    pub const DYN_DATA: [usize; 4] = [140, 168, 196, 224]; // 0x6070..0x6073: type UDINT + 6 REAL
}

/// Bit numbers inside each bit group.
pub mod bit {
    // RobotControl (outputs, 32-bit group at off::ROBOT_CONTROL)
    pub const DEACTIVATE: usize = 0;
    pub const ACTIVATE: usize = 1;
    pub const HOME: usize = 2;
    pub const RESET_ERROR: usize = 3;
    pub const SIM_MODE: usize = 4;
    pub const RECOVERY_MODE: usize = 5;
    pub const DISABLE_ETHERCAT: usize = 6;
    // MotionControl (outputs, 16-bit group at off::MOTION_CONTROL)
    pub const SET_POINT: usize = 0;
    pub const PAUSE: usize = 1;
    pub const CLEAR_MOVE: usize = 2;
    pub const RESUME_MOTION: usize = 3;
    // RobotStatus (inputs, 16-bit group at off::ROBOT_STATUS)
    pub const BUSY: usize = 0;
    pub const ACTIVATED: usize = 1;
    pub const HOMED: usize = 2;
    pub const SIM_ACTIVATED: usize = 3;
    pub const BRAKES_ENGAGED: usize = 4;
    pub const RECOVERY: usize = 5;
    pub const ESTOP: usize = 6;
    pub const COLLISION: usize = 7;
    pub const WORKZONE: usize = 8;
    // MotionStatus (inputs, 16-bit group at off::MOTION_STATUS)
    pub const PAUSED: usize = 0;
    pub const EOB: usize = 1;
    pub const EOM: usize = 2;
    pub const CLEARED: usize = 3;
    pub const PSTOP2: usize = 4;
    pub const EXCESSIVE_TORQUE: usize = 5;
}

/// Motion command ids (0x7305). Activate, Home, ClearMotion... are bits, not commands.
pub mod cmd {
    pub const NONE: u32 = 0;
    pub const MOVE_JOINTS: u32 = 1;
    pub const SET_BLENDING: u32 = 7;
    pub const SET_JOINT_VEL: u32 = 8;
    pub const SET_JOINT_ACC: u32 = 9;
    pub const SET_CART_ACC: u32 = 12;
    pub const MOVE_JOINTS_VEL: u32 = 21;
    pub const MOVE_LIN_VEL_WRF: u32 = 22;
    pub const MOVE_LIN_VEL_TRF: u32 = 23;
    pub const SET_VEL_TIMEOUT: u32 = 24;
    pub const SET_JOINT_VEL_LIMIT: u32 = 33;

    /// Velocity commands run cyclically (MoveID 0): the robot re-applies them every cycle.
    pub fn is_velocity(id: u32) -> bool {
        matches!(id, MOVE_JOINTS_VEL | MOVE_LIN_VEL_WRF | MOVE_LIN_VEL_TRF)
    }
}

/// Dynamic data types of the 4 input slots.
pub mod dyn_type {
    pub const TARGET_JOINT_VEL: u32 = 32;
    pub const TARGET_CART_VEL: u32 = 34;
    pub const JOINT_POS: u32 = 40;
    pub const CART_POS: u32 = 41;
    pub const JOINT_VEL: u32 = 42;
    pub const CART_VEL: u32 = 44;
}

// ------------------------------------------------------------------------------------------- //
// Byte order                                                                                   //
// ------------------------------------------------------------------------------------------- //

/// The programming manual says the cyclic 16/32-bit integers are big-endian, the ESI declares
/// standard (little-endian) EtherCAT types: the order is detected on the robot (detect_order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Le,
    Be,
}

impl ByteOrder {
    pub fn name(self) -> &'static str {
        match self {
            ByteOrder::Le => "le",
            ByteOrder::Be => "be",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Codec {
    pub order: ByteOrder,
}

impl Codec {
    pub fn new(order: ByteOrder) -> Self {
        Codec { order }
    }

    pub fn flipped(&self) -> Codec {
        Codec::new(match self.order {
            ByteOrder::Le => ByteOrder::Be,
            ByteOrder::Be => ByteOrder::Le,
        })
    }

    pub fn get_u16(&self, b: &[u8], o: usize) -> u16 {
        let x = [b[o], b[o + 1]];
        match self.order {
            ByteOrder::Le => u16::from_le_bytes(x),
            ByteOrder::Be => u16::from_be_bytes(x),
        }
    }

    pub fn get_u32(&self, b: &[u8], o: usize) -> u32 {
        let x = [b[o], b[o + 1], b[o + 2], b[o + 3]];
        match self.order {
            ByteOrder::Le => u32::from_le_bytes(x),
            ByteOrder::Be => u32::from_be_bytes(x),
        }
    }

    pub fn get_f32(&self, b: &[u8], o: usize) -> f32 {
        f32::from_bits(self.get_u32(b, o))
    }

    pub fn put_u16(&self, b: &mut [u8], o: usize, v: u16) {
        let x = match self.order {
            ByteOrder::Le => v.to_le_bytes(),
            ByteOrder::Be => v.to_be_bytes(),
        };
        b[o..o + 2].copy_from_slice(&x);
    }

    pub fn put_u32(&self, b: &mut [u8], o: usize, v: u32) {
        let x = match self.order {
            ByteOrder::Le => v.to_le_bytes(),
            ByteOrder::Be => v.to_be_bytes(),
        };
        b[o..o + 4].copy_from_slice(&x);
    }

    pub fn put_f32(&self, b: &mut [u8], o: usize, v: f32) {
        self.put_u32(b, o, v.to_bits());
    }

    /// (byte, mask) of bit `n` of a bit group of `width` bytes at `o`. Little-endian: bit 0 is
    /// in the first byte (the ESI bit offsets). Big-endian: the group read as one integer.
    pub fn bit_pos(&self, o: usize, width: usize, n: usize) -> (usize, u8) {
        let byte = match self.order {
            ByteOrder::Le => o + n / 8,
            ByteOrder::Be => o + width - 1 - n / 8,
        };
        (byte, 1u8 << (n % 8))
    }

    pub fn get_bit(&self, b: &[u8], o: usize, width: usize, n: usize) -> bool {
        let (i, m) = self.bit_pos(o, width, n);
        b[i] & m != 0
    }

    pub fn set_bit(&self, b: &mut [u8], o: usize, width: usize, n: usize, v: bool) {
        let (i, m) = self.bit_pos(o, width, n);
        if v {
            b[i] |= m;
        } else {
            b[i] &= !m;
        }
    }
}

// ------------------------------------------------------------------------------------------- //
// Outputs                                                                                      //
// ------------------------------------------------------------------------------------------- //

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Outputs {
    pub deactivate: bool,
    pub activate: bool,
    pub home: bool,
    pub reset_error: bool,
    pub sim_mode: bool,
    pub recovery_mode: bool,
    pub disable_ethercat: bool,
    pub move_id: u16,
    pub set_point: bool,
    pub pause: bool,
    pub clear_move: bool,
    pub resume_motion: bool,
    pub command_id: u32,
    pub args: [f32; 6],
    pub host_time: u32,
    pub brakes_control_allowed: bool,
    pub brakes_engaged: bool,
    pub dyn_types: [u32; 4],
}

impl Outputs {
    /// `mirror_disable`: while the byte order is not confirmed, DisableEtherCAT is also written
    /// where the other byte order would put it (that byte is padding in the right reading).
    pub fn encode(&self, c: &Codec, buf: &mut [u8; OUT_LEN], mirror_disable: bool) {
        buf.fill(0);
        let rc = [
            self.deactivate,
            self.activate,
            self.home,
            self.reset_error,
            self.sim_mode,
            self.recovery_mode,
            self.disable_ethercat,
        ];
        for (n, &v) in rc.iter().enumerate() {
            c.set_bit(buf, off::ROBOT_CONTROL, 4, n, v);
        }
        if mirror_disable && self.disable_ethercat {
            c.flipped().set_bit(buf, off::ROBOT_CONTROL, 4, bit::DISABLE_ETHERCAT, true);
        }
        c.put_u16(buf, off::MOVE_ID, self.move_id);
        let mc = [self.set_point, self.pause, self.clear_move, self.resume_motion];
        for (n, &v) in mc.iter().enumerate() {
            c.set_bit(buf, off::MOTION_CONTROL, 2, n, v);
        }
        c.put_u32(buf, off::MOVE_COMMAND, self.command_id);
        for i in 0..6 {
            c.put_f32(buf, off::MOVE_ARGS + 4 * i, self.args[i]);
        }
        c.put_u32(buf, off::HOST_TIME, self.host_time);
        c.set_bit(buf, off::BRAKES_CONTROL, 4, 0, self.brakes_control_allowed);
        c.set_bit(buf, off::BRAKES_CONTROL, 4, 1, self.brakes_engaged);
        for i in 0..4 {
            c.put_u32(buf, off::DYN_CFG + 4 * i, self.dyn_types[i]);
        }
    }

    /// Inverse of encode (the simulated robot reads the outputs with it).
    pub fn decode(c: &Codec, b: &[u8; OUT_LEN]) -> Outputs {
        let g = |n| c.get_bit(b, off::ROBOT_CONTROL, 4, n);
        let m = |n| c.get_bit(b, off::MOTION_CONTROL, 2, n);
        let mut args = [0f32; 6];
        for (i, a) in args.iter_mut().enumerate() {
            *a = c.get_f32(b, off::MOVE_ARGS + 4 * i);
        }
        let mut dyn_types = [0u32; 4];
        for (i, t) in dyn_types.iter_mut().enumerate() {
            *t = c.get_u32(b, off::DYN_CFG + 4 * i);
        }
        Outputs {
            deactivate: g(bit::DEACTIVATE),
            activate: g(bit::ACTIVATE),
            home: g(bit::HOME),
            reset_error: g(bit::RESET_ERROR),
            sim_mode: g(bit::SIM_MODE),
            recovery_mode: g(bit::RECOVERY_MODE),
            disable_ethercat: g(bit::DISABLE_ETHERCAT),
            move_id: c.get_u16(b, off::MOVE_ID),
            set_point: m(bit::SET_POINT),
            pause: m(bit::PAUSE),
            clear_move: m(bit::CLEAR_MOVE),
            resume_motion: m(bit::RESUME_MOTION),
            command_id: c.get_u32(b, off::MOVE_COMMAND),
            args,
            host_time: c.get_u32(b, off::HOST_TIME),
            brakes_control_allowed: c.get_bit(b, off::BRAKES_CONTROL, 4, 0),
            brakes_engaged: c.get_bit(b, off::BRAKES_CONTROL, 4, 1),
            dyn_types,
        }
    }

    /// Velocity command on all joints (MoveJointsVel, cyclic: MoveID 0, SetPoint 1).
    pub fn joint_vel(&mut self, v: &[f64; 6]) {
        self.command_id = cmd::MOVE_JOINTS_VEL;
        self.move_id = 0;
        self.set_point = true;
        for i in 0..6 {
            self.args[i] = v[i] as f32;
        }
    }

    /// A queued command (MoveJoints, Set...): new MoveID, held until the robot echoes it.
    pub fn queued(&mut self, id: u32, args: &[f64; 6], move_id: u16) {
        self.command_id = id;
        self.move_id = move_id;
        self.set_point = true;
        for i in 0..6 {
            self.args[i] = args[i] as f32;
        }
    }

    /// If a velocity command is running, make it zero (nothing else changes: no new command).
    pub fn zero_velocity(&mut self) {
        if cmd::is_velocity(self.command_id) && self.move_id == 0 {
            self.args = [0.0; 6];
        }
    }
}

// ------------------------------------------------------------------------------------------- //
// Inputs                                                                                       //
// ------------------------------------------------------------------------------------------- //

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Inputs {
    pub busy: bool,
    pub activated: bool,
    pub homed: bool,
    pub sim_activated: bool,
    pub brakes_engaged: bool,
    pub recovery_mode: bool,
    pub estop: bool,
    pub collision: bool,
    pub workzone: bool,
    pub error_code: u16,
    pub checkpoint_reached: u16,
    pub checkpoint_discarded: u16,
    pub move_id: u16,
    pub fifo_space: u16,
    pub paused: bool,
    pub eob: bool,
    pub eom: bool,
    pub cleared: bool,
    pub pstop2: bool,
    pub excessive_torque: bool,
    pub offline_program_id: u16,
    pub joint_set: [f64; 6],
    pub pose: [f64; 6],
    pub ts_s: u32,
    pub ts_us: u32,
    pub dyn_cycles: u32,
    pub safety: [u8; 12],
    pub dyn_type: [u32; 4],
    pub dyn_val: [[f64; 6]; 4],
}

impl Inputs {
    pub fn decode(c: &Codec, b: &[u8; IN_LEN]) -> Inputs {
        let s = |n| c.get_bit(b, off::ROBOT_STATUS, 2, n);
        let m = |n| c.get_bit(b, off::MOTION_STATUS, 2, n);
        let six = |o: usize| {
            let mut v = [0f64; 6];
            for (i, x) in v.iter_mut().enumerate() {
                *x = c.get_f32(b, o + 4 * i) as f64;
            }
            v
        };
        let mut dyn_type = [0u32; 4];
        let mut dyn_val = [[0f64; 6]; 4];
        for k in 0..4 {
            dyn_type[k] = c.get_u32(b, off::DYN_DATA[k]);
            dyn_val[k] = six(off::DYN_DATA[k] + 4);
        }
        let mut safety = [0u8; 12];
        safety.copy_from_slice(&b[off::SAFETY..off::SAFETY + 12]);
        Inputs {
            busy: s(bit::BUSY),
            activated: s(bit::ACTIVATED),
            homed: s(bit::HOMED),
            sim_activated: s(bit::SIM_ACTIVATED),
            brakes_engaged: s(bit::BRAKES_ENGAGED),
            recovery_mode: s(bit::RECOVERY),
            estop: s(bit::ESTOP),
            collision: s(bit::COLLISION),
            workzone: s(bit::WORKZONE),
            error_code: c.get_u16(b, off::ERROR_CODE),
            checkpoint_reached: c.get_u16(b, off::CHECKPOINT_REACHED),
            checkpoint_discarded: c.get_u16(b, off::CHECKPOINT_DISCARDED),
            move_id: c.get_u16(b, off::MOVE_ID_ECHO),
            fifo_space: c.get_u16(b, off::FIFO_SPACE),
            paused: m(bit::PAUSED),
            eob: m(bit::EOB),
            eom: m(bit::EOM),
            cleared: m(bit::CLEARED),
            pstop2: m(bit::PSTOP2),
            excessive_torque: m(bit::EXCESSIVE_TORQUE),
            offline_program_id: c.get_u16(b, off::OFFLINE_PROGRAM_ID),
            joint_set: six(off::JOINT_SET),
            pose: six(off::POSE),
            ts_s: c.get_u32(b, off::TIMESTAMP_S),
            ts_us: c.get_u32(b, off::TIMESTAMP_US),
            dyn_cycles: c.get_u32(b, off::DYN_CYCLES),
            safety,
            dyn_type,
            dyn_val,
        }
    }

    /// Inverse of decode (the simulated robot writes its inputs with it).
    pub fn encode(&self, c: &Codec, b: &mut [u8; IN_LEN]) {
        b.fill(0);
        let st = [
            self.busy,
            self.activated,
            self.homed,
            self.sim_activated,
            self.brakes_engaged,
            self.recovery_mode,
            self.estop,
            self.collision,
            self.workzone,
        ];
        for (n, &v) in st.iter().enumerate() {
            c.set_bit(b, off::ROBOT_STATUS, 2, n, v);
        }
        c.put_u16(b, off::ERROR_CODE, self.error_code);
        c.put_u16(b, off::CHECKPOINT_REACHED, self.checkpoint_reached);
        c.put_u16(b, off::CHECKPOINT_DISCARDED, self.checkpoint_discarded);
        c.put_u16(b, off::MOVE_ID_ECHO, self.move_id);
        c.put_u16(b, off::FIFO_SPACE, self.fifo_space);
        let ms = [self.paused, self.eob, self.eom, self.cleared, self.pstop2, self.excessive_torque];
        for (n, &v) in ms.iter().enumerate() {
            c.set_bit(b, off::MOTION_STATUS, 2, n, v);
        }
        c.put_u16(b, off::OFFLINE_PROGRAM_ID, self.offline_program_id);
        for i in 0..6 {
            c.put_f32(b, off::JOINT_SET + 4 * i, self.joint_set[i] as f32);
            c.put_f32(b, off::POSE + 4 * i, self.pose[i] as f32);
        }
        c.put_u32(b, off::TIMESTAMP_S, self.ts_s);
        c.put_u32(b, off::TIMESTAMP_US, self.ts_us);
        c.put_u32(b, off::DYN_CYCLES, self.dyn_cycles);
        b[off::SAFETY..off::SAFETY + 12].copy_from_slice(&self.safety);
        for k in 0..4 {
            c.put_u32(b, off::DYN_DATA[k], self.dyn_type[k]);
            for i in 0..6 {
                c.put_f32(b, off::DYN_DATA[k] + 4 + 4 * i, self.dyn_val[k][i] as f32);
            }
        }
    }

    /// Robot timestamp [s] (seconds + microseconds, monotonic from an arbitrary origin).
    pub fn robot_time(&self) -> f64 {
        self.ts_s as f64 + self.ts_us as f64 * 1e-6
    }

    /// The values of a dynamic data type, if one of the 4 slots carries it this cycle.
    pub fn dyn_of(&self, ty: u32) -> Option<[f64; 6]> {
        (0..4).find(|&k| self.dyn_type[k] == ty).map(|k| self.dyn_val[k])
    }

    pub fn status_line(&self) -> String {
        let f = |b: bool, s: &'static str| if b { s } else { "" };
        format!(
            "act {} homed {} err {} busy {} paused {} eob {} eom {} estop {} sim {} | move id {} fifo {} | t {:.6} s",
            self.activated as u8,
            self.homed as u8,
            self.error_code,
            self.busy as u8,
            self.paused as u8,
            self.eob as u8,
            self.eom as u8,
            self.estop as u8,
            self.sim_activated as u8,
            self.move_id,
            self.fifo_space,
            self.robot_time()
        ) + f(self.cleared, " cleared")
    }
}

// ------------------------------------------------------------------------------------------- //
// Frame row: the 54 columns of meca_latency_test.py (see its Recorder)                         //
// ------------------------------------------------------------------------------------------- //

pub const NCOL: usize = 6 + 8 * NJ;
pub const HOST: usize = 0;
pub const ROBOT: usize = 1;
pub const TPOS: usize = 2;
pub const TVEL: usize = 3;
pub const APOS: usize = 4;
pub const AVEL: usize = 5;
pub const K_TPOS: usize = 0;
pub const K_TVEL: usize = 1;
pub const K_APOS: usize = 2;
pub const K_AVEL: usize = 3;
pub const K_TCART: usize = 4;
pub const K_TCVEL: usize = 5;
pub const K_ACART: usize = 6;
pub const K_ACVEL: usize = 7;

pub type Row = [f64; NCOL];

pub fn col(kind: usize, j: usize) -> usize {
    6 + kind * NJ + j
}

pub fn alljoints(row: &Row, kind: usize) -> [f64; 6] {
    let mut v = [0.0; 6];
    v.copy_from_slice(&row[col(kind, 0)..col(kind, 0) + 6]);
    v
}

/// One frame row: host receive time, robot time, the tested joint `j` (0-based) twice, then the
/// 8 signal groups of every joint. A signal no slot carries this cycle is NaN.
pub fn frame_row(inp: &Inputs, host: f64, j: usize) -> Row {
    let nan = [f64::NAN; 6];
    let groups = [
        inp.joint_set,
        inp.dyn_of(dyn_type::TARGET_JOINT_VEL).unwrap_or(nan),
        inp.dyn_of(dyn_type::JOINT_POS).unwrap_or(nan),
        inp.dyn_of(dyn_type::JOINT_VEL).unwrap_or(nan),
        inp.pose,
        inp.dyn_of(dyn_type::TARGET_CART_VEL).unwrap_or(nan),
        inp.dyn_of(dyn_type::CART_POS).unwrap_or(nan),
        inp.dyn_of(dyn_type::CART_VEL).unwrap_or(nan),
    ];
    let mut r = [f64::NAN; NCOL];
    r[HOST] = host;
    r[ROBOT] = inp.robot_time();
    for (k, g) in groups.iter().enumerate() {
        r[col(k, 0)..col(k, 0) + 6].copy_from_slice(g);
    }
    r[TPOS] = groups[K_TPOS][j];
    r[TVEL] = groups[K_TVEL][j];
    r[APOS] = groups[K_APOS][j];
    r[AVEL] = groups[K_AVEL][j];
    r
}

// ------------------------------------------------------------------------------------------- //
// Byte order detection                                                                         //
// ------------------------------------------------------------------------------------------- //

/// Is this reading of the input samples plausible? Microseconds below 1e6, a robot clock that
/// moves forward by small steps, finite joint angles within +/-1000 deg.
fn plausible(c: &Codec, samples: &[[u8; IN_LEN]]) -> bool {
    let mut last: Option<f64> = None;
    let mut steps = 0;
    for b in samples {
        let us = c.get_u32(b, off::TIMESTAMP_US);
        if us >= 1_000_000 {
            return false;
        }
        let t = c.get_u32(b, off::TIMESTAMP_S) as f64 + us as f64 * 1e-6;
        for j in 0..6 {
            let q = c.get_f32(b, off::JOINT_SET + 4 * j);
            if !q.is_finite() || q.abs() > 1000.0 {
                return false;
            }
        }
        if let Some(l) = last {
            if t < l || t - l > 0.5 {
                return false;
            }
            if t > l {
                steps += 1;
            }
        }
        last = Some(t);
    }
    steps >= 3
}

/// Byte order of the cyclic data, from a few dozen input images taken while nothing moves.
pub fn detect_order(samples: &[[u8; IN_LEN]]) -> Result<ByteOrder, String> {
    let le = plausible(&Codec::new(ByteOrder::Le), samples);
    let be = plausible(&Codec::new(ByteOrder::Be), samples);
    match (le, be) {
        (true, false) => Ok(ByteOrder::Le),
        (false, true) => Ok(ByteOrder::Be),
        (true, true) => Err("both byte orders look plausible (robot clock not moving?)".into()),
        (false, false) => Err(format!(
            "no plausible byte order in {} input images (robot timestamp / joint angles)",
            samples.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn esi_offsets() -> Vec<(String, u32, u32, usize, usize)> {
        // (kind, index, subindex, bit offset, bit length) of every PDO entry, from the ESI
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/esi/Meca500_EtherCAT_ESI_v11.3.1.11942-official.xml");
        let text = std::fs::read_to_string(path).expect("ESI file");
        let doc = roxmltree::Document::parse(&text).expect("ESI xml");
        let num = |s: &str| -> u32 {
            let s = s.trim();
            if let Some(h) = s.strip_prefix("#x") {
                u32::from_str_radix(h, 16).unwrap()
            } else {
                s.parse().unwrap()
            }
        };
        let dev = doc.descendants().find(|n| n.has_tag_name("Device")).unwrap();
        let mut out = Vec::new();
        for kind in ["RxPdo", "TxPdo"] {
            let mut bitoff = 0usize;
            for pdo in dev.children().filter(|n| n.has_tag_name(kind)) {
                for e in pdo.children().filter(|n| n.has_tag_name("Entry")) {
                    let get = |t: &str| e.children().find(|n| n.has_tag_name(t)).and_then(|n| n.text());
                    let idx = num(get("Index").unwrap());
                    let sub = get("SubIndex").map(num).unwrap_or(0);
                    let len = num(get("BitLen").unwrap()) as usize;
                    out.push((kind.to_string(), idx, sub, bitoff, len));
                    bitoff += len;
                }
            }
            let total: usize = out.iter().filter(|x| x.0 == kind).map(|x| x.4).sum();
            assert_eq!(total, if kind == "RxPdo" { OUT_LEN * 8 } else { IN_LEN * 8 }, "{kind} size");
        }
        out
    }

    #[test]
    fn offsets_match_the_esi() {
        let e = esi_offsets();
        let at = |kind: &str, idx: u32, sub: u32| -> usize {
            e.iter().find(|x| x.0 == kind && x.1 == idx && x.2 == sub).unwrap_or_else(|| panic!("{idx:#x}:{sub}")).3
        };
        let rx = |idx, sub| at("RxPdo", idx, sub);
        let tx = |idx, sub| at("TxPdo", idx, sub);
        assert_eq!(rx(0x7200, 1), off::ROBOT_CONTROL * 8 + bit::DEACTIVATE);
        assert_eq!(rx(0x7200, 2), off::ROBOT_CONTROL * 8 + bit::ACTIVATE);
        assert_eq!(rx(0x7200, 3), off::ROBOT_CONTROL * 8 + bit::HOME);
        assert_eq!(rx(0x7200, 4), off::ROBOT_CONTROL * 8 + bit::RESET_ERROR);
        assert_eq!(rx(0x7200, 5), off::ROBOT_CONTROL * 8 + bit::SIM_MODE);
        assert_eq!(rx(0x7200, 7), off::ROBOT_CONTROL * 8 + bit::DISABLE_ETHERCAT);
        assert_eq!(rx(0x7310, 1), off::MOVE_ID * 8);
        assert_eq!(rx(0x7310, 2), off::MOTION_CONTROL * 8 + bit::SET_POINT);
        assert_eq!(rx(0x7310, 3), off::MOTION_CONTROL * 8 + bit::PAUSE);
        assert_eq!(rx(0x7310, 4), off::MOTION_CONTROL * 8 + bit::CLEAR_MOVE);
        assert_eq!(rx(0x7310, 5), off::MOTION_CONTROL * 8 + bit::RESUME_MOTION);
        assert_eq!(rx(0x7305, 0), off::MOVE_COMMAND * 8);
        for i in 0..6 {
            assert_eq!(rx(0x7306, i + 1), (off::MOVE_ARGS + 4 * i as usize) * 8);
        }
        assert_eq!(rx(0x7400, 0), off::HOST_TIME * 8);
        assert_eq!(rx(0x7410, 1), off::BRAKES_CONTROL * 8);
        for k in 0..4u32 {
            assert_eq!(rx(0x7420 + k, 1), (off::DYN_CFG + 4 * k as usize) * 8);
        }
        assert_eq!(tx(0x6010, 2), off::ROBOT_STATUS * 8 + bit::BUSY);
        assert_eq!(tx(0x6010, 3), off::ROBOT_STATUS * 8 + bit::ACTIVATED);
        assert_eq!(tx(0x6010, 4), off::ROBOT_STATUS * 8 + bit::HOMED);
        assert_eq!(tx(0x6010, 5), off::ROBOT_STATUS * 8 + bit::SIM_ACTIVATED);
        assert_eq!(tx(0x6010, 8), off::ROBOT_STATUS * 8 + bit::ESTOP);
        assert_eq!(tx(0x6010, 10), off::ROBOT_STATUS * 8 + bit::WORKZONE);
        assert_eq!(tx(0x6010, 1), off::ERROR_CODE * 8);
        assert_eq!(tx(0x6015, 1), off::CHECKPOINT_REACHED * 8);
        assert_eq!(tx(0x6015, 3), off::MOVE_ID_ECHO * 8);
        assert_eq!(tx(0x6015, 4), off::FIFO_SPACE * 8);
        assert_eq!(tx(0x6015, 6), off::MOTION_STATUS * 8 + bit::PAUSED);
        assert_eq!(tx(0x6015, 7), off::MOTION_STATUS * 8 + bit::EOB);
        assert_eq!(tx(0x6015, 8), off::MOTION_STATUS * 8 + bit::EOM);
        assert_eq!(tx(0x6015, 9), off::MOTION_STATUS * 8 + bit::CLEARED);
        assert_eq!(tx(0x6015, 5), off::OFFLINE_PROGRAM_ID * 8);
        assert_eq!(tx(0x6030, 1), off::JOINT_SET * 8);
        assert_eq!(tx(0x6031, 1), off::POSE * 8);
        assert_eq!(tx(0x6060, 1), off::TIMESTAMP_S * 8);
        assert_eq!(tx(0x6060, 2), off::TIMESTAMP_US * 8);
        assert_eq!(tx(0x6060, 3), off::DYN_CYCLES * 8);
        assert_eq!(tx(0x6065, 1), off::SAFETY * 8);
        for k in 0..4u32 {
            assert_eq!(tx(0x6070 + k, 1), off::DYN_DATA[k as usize] * 8);
            assert_eq!(tx(0x6070 + k, 2), (off::DYN_DATA[k as usize] + 4) * 8);
        }
    }

    #[test]
    fn outputs_round_trip_both_orders() {
        for order in [ByteOrder::Le, ByteOrder::Be] {
            let c = Codec::new(order);
            let o = Outputs {
                activate: true,
                disable_ethercat: true,
                move_id: 0x1234,
                set_point: true,
                resume_motion: true,
                command_id: cmd::MOVE_JOINTS_VEL,
                args: [1.5, -2.0, 0.0, 3.25, 100.0, -0.125],
                dyn_types: [32, 40, 42, 41],
                ..Default::default()
            };
            let mut b = [0u8; OUT_LEN];
            o.encode(&c, &mut b, false);
            assert_eq!(Outputs::decode(&c, &b), o);
        }
    }

    #[test]
    fn little_endian_layout_is_the_esi_one() {
        let c = Codec::new(ByteOrder::Le);
        let o = Outputs { disable_ethercat: true, move_id: 0x0102, set_point: true, command_id: 21, ..Default::default() };
        let mut b = [0u8; OUT_LEN];
        o.encode(&c, &mut b, false);
        assert_eq!(b[0], 0x40); // DisableEtherCAT = byte 0 bit 6 ("0:6" in the manual)
        assert_eq!(&b[4..6], &[0x02, 0x01]);
        assert_eq!(b[6], 0x01);
        assert_eq!(&b[8..12], &[21, 0, 0, 0]);
        let mut m = [0u8; OUT_LEN];
        o.encode(&c, &mut m, true); // mirrored: also byte 3 bit 6 (big-endian reading)
        assert_eq!((m[0], m[3]), (0x40, 0x40));
    }

    #[test]
    fn inputs_round_trip_and_frame_row() {
        let c = Codec::new(ByteOrder::Be);
        let mut i = Inputs { activated: true, homed: true, eob: true, error_code: 1005, move_id: 77, ..Default::default() };
        i.joint_set = [1.0, 2.0, 3.0, 4.0, 5.0, 6.5];
        i.ts_s = 1234;
        i.ts_us = 567_000;
        i.dyn_type = [32, 40, 42, 0];
        i.dyn_val[0] = [0.5; 6];
        i.dyn_val[1] = [7.0; 6];
        let mut b = [0u8; IN_LEN];
        i.encode(&c, &mut b);
        let d = Inputs::decode(&c, &b);
        assert_eq!(d, i);
        let r = frame_row(&d, 10.0, 5);
        assert_eq!((r[HOST], r[ROBOT]), (10.0, 1234.567));
        assert_eq!((r[TPOS], r[TVEL], r[APOS]), (6.5, 0.5, 7.0));
        assert!(r[AVEL].is_nan());
        assert_eq!(r[col(K_TPOS, 0)], 1.0);
        assert!(r[col(K_ACART, 0)].is_nan());
    }

    #[test]
    fn byte_order_detection() {
        for order in [ByteOrder::Le, ByteOrder::Be] {
            let c = Codec::new(order);
            let samples: Vec<[u8; IN_LEN]> = (0..40)
                .map(|k| {
                    let mut i = Inputs::default();
                    i.ts_s = 5000;
                    i.ts_us = 123_456 + 1000 * (k / 2); // a new sample every other cycle
                    i.joint_set = [0.0, 10.0, -20.0, 0.5, 30.0, 90.0];
                    let mut b = [0u8; IN_LEN];
                    i.encode(&c, &mut b);
                    b
                })
                .collect();
            assert_eq!(detect_order(&samples), Ok(order));
        }
        let still = vec![[0u8; IN_LEN]; 10];
        assert!(detect_order(&still).is_err());
    }
}
