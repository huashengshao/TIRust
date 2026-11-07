//! Kinematics and helpers shared by the store, rotation-copy, concurrency
//! and amplitude-sharing tests: four physical hexagon points, a small
//! deterministic generator and a bitwise fingerprint of a batch result.
use iregi_masters::NativeFloat;
use iregi_rust::BatchResult;

/// Four top-loop gg -> gggg hexagon points of the binary64 stability study
/// (sqrt(s) = 1 TeV; shifts of the six propagators), m_t^2 = 29929, mu = 1000.
pub const POINTS: [[[f64; 4]; 6]; 4] = [
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [792.7874729156998, -82.84167294384719, -92.33145751079824, -165.97948836665938],
        [496.1459371514851, -309.53730217508144, -274.5389438821032, -224.33807246752178],
        [365.75699134826516, -254.15872221430251, -237.93327663297802, -112.1127847893476],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [628.3485871137328, -126.38562954754646, -64.66171687618484, 343.46805923528484],
        [493.6035005043037, -75.99778103019892, 59.08875070855528, 360.8785405535477],
        [342.2594345535183, -151.0438342019983, 151.3291327069494, 267.2578797427977],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [720.5411460864439, 41.03231623557652, 269.0508926250419, -63.44459971258532],
        [580.5195723069281, 86.55172419413077, 347.3583949823081, 43.335359537087676],
        [328.9555327352336, -146.2746278878062, 292.60284001366847, -34.62735579481515],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [684.7727360619465, -266.1383293706448, -32.469872857286525, 165.7839706555041],
        [491.59350180547904, -99.14409525080174, -121.72713563298282, 127.51801502347416],
        [243.28769056900498, 107.6510206808816, -17.784570933495488, 217.44853912012346],
    ],
];
pub const MT2: f64 = 29929.0;
pub const MU: f64 = 1000.0;

/// A small deterministic generator (the lanes must not depend on a crate).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Every lane's array (best-effort ones included) and status, exactly (the
/// shortest round-trip forms of every limb).
pub fn fingerprint<T: NativeFloat>(result: &BatchResult<T>) -> (Vec<String>, String) {
    let values = (0..result.rows())
        .map(|lane| result.lane_with_estimate(lane).map_or_else(|| "refused".to_string(), |best| format!("{:?}", best.values)))
        .collect();
    (values, format!("{:?}", result.statuses))
}
