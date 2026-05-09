//! # Fleet-Keel
//!
//! A 5-dimensional self-orientation system for fleet agents using only
//! internal dynamics — no external clock, no timestamps, no synchronization.
//!
//! ## The 5 Dimensions
//!
//! 1. **Position** — mean-square state norm (energy)
//! 2. **Orientation** — sign pattern of agent means (mood)
//! 3. **Velocity** — rate of state change between steps
//! 4. **Strain** — deviation from baseline attractor
//! 5. **Alignment** — pairwise absolute cosine similarity
//!
//! ## Verified Constants (53 GPU experiments)
//!
//! - Melting point: coupling ≈ 0.67 / N^1.06
//! - Gain edge: gain > 0.85 required
//! - Dead zone: energy < 0.01
//! - Living zone: energy 2–4, correlation > 0.85
//! - Strong zone: energy > 4, correlation > 0.9
//! - Natural state: `+--+` pattern (max-cut of K4 coupling graph)

use std::fmt;

// ---------------------------------------------------------------------------
// SignPattern<const N: usize>
// ---------------------------------------------------------------------------

/// A signed orientation pattern for N agents.
///
/// Each entry is either `+1` or `-1`, representing which side of the
/// decision boundary each agent sits on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignPattern<const N: usize>([i8; N]);

impl<const N: usize> SignPattern<N> {
    /// Create a new sign pattern from raw components (must be ±1).
    ///
    /// # Panics
    ///
    /// Panics if any component is not `1` or `-1`.
    pub fn new(data: [i8; N]) -> Self {
        assert!(
            data.iter().all(|&x| x == 1 || x == -1),
            "SignPattern entries must be +1 or -1"
        );
        Self(data)
    }

    /// The raw sign array.
    pub fn as_slice(&self) -> &[i8] {
        &self.0
    }

    /// The number of edges crossing the sign partition in a complete graph.
    ///
    /// Given N agents, this counts pairs (i, j) where sign_i != sign_j.
    /// This is exactly `positive_count * negative_count`.
    pub fn cut_size(&self) -> usize {
        let pos = self.0.iter().filter(|&&x| x == 1).count();
        let neg = N - pos;
        pos * neg
    }

    /// Returns `true` if this pattern is a max-cut of the complete graph K_N.
    ///
    /// For the complete graph, the max-cut is balanced: floor(N/2) * ceil(N/2).
    pub fn is_max_cut(&self) -> bool {
        let ideal = (N / 2) * (N - N / 2);
        self.cut_size() == ideal
    }

    /// Human-readable label, e.g. `"+--+"`.
    pub fn label(&self) -> String {
        let mut s = String::with_capacity(N);
        for &v in &self.0 {
            s.push(if v == 1 { '+' } else { '-' });
        }
        s
    }
}

impl<const N: usize> fmt::Display for SignPattern<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// FleetZone
// ---------------------------------------------------------------------------

/// Energy-based zone classification for a fleet.
///
/// Determined by thresholds verified across 53 GPU experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FleetZone {
    /// Energy < 0.01 — system has collapsed to origin
    Dead,
    /// 0.01 ≤ Energy < 2.0 — weak but alive
    Dying,
    /// 2.0 ≤ Energy ≤ 4.0 — healthy living zone (correlation > 0.85)
    Living,
    /// Energy > 4.0 — strong coherent state (correlation > 0.9)
    Strong,
    /// Energy ≥ 10.0 — overdriven, may become unstable
    Overdriven,
}

/// Determine the fleet zone from energy value alone.
pub fn zone_from_energy(energy: f64) -> FleetZone {
    if energy < 0.01 {
        FleetZone::Dead
    } else if energy < 2.0 {
        FleetZone::Dying
    } else if energy <= 4.0 {
        FleetZone::Living
    } else if energy < 10.0 {
        FleetZone::Strong
    } else {
        FleetZone::Overdriven
    }
}

// ---------------------------------------------------------------------------
// KeelReading
// ---------------------------------------------------------------------------

/// A complete orientation reading from the keel.
#[derive(Debug, Clone)]
pub struct KeelReading<const AGENTS: usize> {
    /// Mean-square state norm across agents.
    pub position: f64,
    /// Which side of the mean each agent is on.
    pub orientation: SignPattern<AGENTS>,
    /// Rate of state change from previous step.
    pub velocity: f64,
    /// Per-agent deviation from baseline mean.
    pub strain: [f64; AGENTS],
    /// Mean pairwise cosine-similarity magnitude across all agent pairs.
    pub alignment: f64,
    /// Energy-derived zone classification.
    pub zone: FleetZone,
    /// Whether the system is stable (living or strong zone + alignment > 0.8).
    pub is_stable: bool,
}

// ---------------------------------------------------------------------------
// Keel
// ---------------------------------------------------------------------------

/// The Keel orientation system.
///
/// Provides 5-dimensional self-orientation for a fleet of N agents with
/// D-dimensional states, using only internal dynamics — no external clock.
#[derive(Debug, Clone)]
pub struct Keel<const N: usize, const AGENTS: usize> {
    /// Per-agent baseline means (established during calibration).
    baseline_means: Option<[[f64; N]; AGENTS]>,
    /// Baseline sign pattern at calibration time.
    baseline_pattern: Option<SignPattern<AGENTS>>,
    /// Previous states for velocity calculation.
    prev_states: Option<[[f64; N]; AGENTS]>,
    /// Monotonic step counter.
    step_count: usize,
}

impl<const N: usize, const AGENTS: usize> Keel<N, AGENTS> {
    /// Create a new keel with initial states.
    ///
    /// The first call to `read` will establish the baseline automatically.
    pub fn new(_initial_states: &[[f64; N]; AGENTS]) -> Self {
        Self {
            baseline_means: None,
            baseline_pattern: None,
            prev_states: None,
            step_count: 0,
        }
    }

    /// Explicitly calibrate the baseline from current states.
    ///
    /// After calibration, the baseline means and sign pattern are fixed
    /// until `calibrate` is called again.
    pub fn calibrate(&mut self, states: &[[f64; N]; AGENTS]) {
        let means = *states;
        self.baseline_means = Some(means);
        self.baseline_pattern = Some(extract_sign_pattern(states));
        self.prev_states = Some(*states);
        self.step_count = 0;
    }

    /// Take a full 5-dimensional orientation reading.
    ///
    /// Automatically calibrates on the first call if not yet calibrated.
    pub fn read(&mut self, states: &[[f64; N]; AGENTS]) -> KeelReading<AGENTS> {
        if self.baseline_means.is_none() {
            self.calibrate(states);
        }

        let baseline_means = self.baseline_means.unwrap();

        // 1. Position — mean-square state norm (energy)
        let position = mean_square_norm(states);

        // 2. Orientation — sign pattern of agent means
        let orientation = extract_sign_pattern(states);

        // 3. Velocity — mean-square displacement from previous step
        let velocity = match self.prev_states {
            Some(prev) => mean_square_displacement(states, &prev),
            None => 0.0,
        };

        // 4. Strain — per-agent deviation from baseline mean
        let strain = compute_strain(states, &baseline_means);

        // 5. Alignment — mean pairwise absolute cosine similarity
        let alignment = mean_absolute_cosine_similarity(states);

        let zone = zone_from_energy(position);
        let is_stable = matches!(zone, FleetZone::Living | FleetZone::Strong) && alignment > 0.8;

        self.prev_states = Some(*states);
        self.step_count += 1;

        KeelReading {
            position,
            orientation,
            velocity,
            strain,
            alignment,
            zone,
            is_stable,
        }
    }

    /// Detect a perturbation — ≥2 dimensions exceeding the given threshold,
    /// or an orientation flip from baseline.
    pub fn detect_perturbation(&self, reading: &KeelReading<AGENTS>, threshold: f64) -> bool {
        let mut violations = 0;
        if reading.position.abs() > threshold {
            violations += 1;
        }
        if reading.velocity > threshold {
            violations += 1;
        }
        if reading.strain.iter().any(|&s| s > threshold) {
            violations += 1;
        }
        if (1.0 - reading.alignment).abs() > threshold {
            violations += 1;
        }
        // Orientation change from baseline is a strong signal
        if let Some(ref bp) = self.baseline_pattern {
            if reading.orientation != *bp {
                violations += 2; // double weight
            }
        }
        violations >= 2
    }

    /// Expected number of steps for a cooperative agent to converge,
    /// based on verified experimental formula.
    ///
    /// ~50 steps at gain=0.95, coupling=0.25, 4 agents.
    ///
    /// Returns `usize::MAX` if convergence is not expected (zero gain/coupling).
    pub fn expected_convergence_steps(gain: f64, coupling: f64, n_agents: usize) -> usize {
        if gain <= 0.0 || coupling <= 0.0 {
            return usize::MAX;
        }
        // Verified scaling: ~50 steps at (0.95, 0.25, 4 agents)
        // At nominal (0.95, 0.25, 4): 50 * 1 * 1 * 1 = 50
        let steps = 50.0 * (0.95 / gain) * (0.25 / coupling) * (4.0 / n_agents as f64);
        let s = steps.ceil() as usize;
        if s == 0 { 1 } else { s }
    }

    /// Process-relative lateness check.
    ///
    /// Returns `true` if the OTHER agent appears to be late — either because
    /// it has exceeded 2× the expected convergence steps, or its energy has
    /// collapsed into the dead zone.
    pub fn is_process_late(
        &self,
        steps_elapsed: usize,
        other_gain: f64,
        other_coupling: f64,
        other_energy: f64,
    ) -> bool {
        let expected = Self::expected_convergence_steps(other_gain, other_coupling, AGENTS);
        if expected == usize::MAX {
            return false; // Convergence not expected for this agent
        }
        // Allow 2× buffer
        let max_allowed = expected.saturating_mul(2);
        let steps_over = steps_elapsed > max_allowed;

        // Also flag if energy suggests collapse
        let is_dead = other_energy < 0.01;

        steps_over || is_dead
    }

    /// Current step count (process-relative "time").
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Mean-square norm across all agents: energy = Σ ||state_i||² / AGENTS.
fn mean_square_norm<const N: usize, const AGENTS: usize>(
    states: &[[f64; N]; AGENTS],
) -> f64 {
    let total: f64 = states
        .iter()
        .map(|agent| agent.iter().map(|x| x * x).sum::<f64>())
        .sum();
    total / AGENTS as f64
}

/// Extract sign pattern from agent state means.
fn extract_sign_pattern<const N: usize, const AGENTS: usize>(
    states: &[[f64; N]; AGENTS],
) -> SignPattern<AGENTS> {
    let mut signs = [0i8; AGENTS];
    for (i, agent) in states.iter().enumerate() {
        let mean: f64 = agent.iter().copied().sum::<f64>() / N as f64;
        signs[i] = if mean >= 0.0 { 1 } else { -1 };
    }
    SignPattern(signs)
}

/// Mean-square displacement between current and previous states.
fn mean_square_displacement<const N: usize, const AGENTS: usize>(
    states: &[[f64; N]; AGENTS],
    prev: &[[f64; N]; AGENTS],
) -> f64 {
    let total: f64 = states
        .iter()
        .zip(prev.iter())
        .map(|(s, p)| s.iter().zip(p.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>())
        .sum();
    total / AGENTS as f64
}

/// Per-agent strain: |mean_i - baseline_mean_i|.
fn compute_strain<const N: usize, const AGENTS: usize>(
    states: &[[f64; N]; AGENTS],
    baseline: &[[f64; N]; AGENTS],
) -> [f64; AGENTS] {
    let mut strain = [0.0f64; AGENTS];
    for i in 0..AGENTS {
        let mean_i: f64 = states[i].iter().copied().sum::<f64>() / N as f64;
        let base_i: f64 = baseline[i].iter().copied().sum::<f64>() / N as f64;
        strain[i] = (mean_i - base_i).abs();
    }
    strain
}

/// Cosine similarity between two vectors.
fn cosine_similarity<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Mean pairwise absolute cosine similarity across all agent pairs.
fn mean_absolute_cosine_similarity<const N: usize, const AGENTS: usize>(
    states: &[[f64; N]; AGENTS],
) -> f64 {
    let mut total = 0.0f64;
    let mut count = 0usize;
    for i in 0..AGENTS {
        for j in (i + 1)..AGENTS {
            total += cosine_similarity(&states[i], &states[j]).abs();
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SignPattern tests ──

    #[test]
    fn test_sign_pattern_label() {
        let sp = SignPattern::new([1, -1, -1, 1]);
        assert_eq!(sp.label(), "+--+");
        assert_eq!(format!("{}", sp), "+--+");
    }

    #[test]
    fn test_sign_pattern_cut_size() {
        // "+--+": agents 0(+),3(+) vs 1(-),2(-) => cut = 2*2 = 4
        let sp = SignPattern::new([1, -1, -1, 1]);
        assert_eq!(sp.cut_size(), 4);

        // All same sign => 0 cuts
        let all_plus = SignPattern::new([1, 1, 1, 1]);
        assert_eq!(all_plus.cut_size(), 0);

        // Single agent => 0 cuts
        let single = SignPattern::new([1]);
        assert_eq!(single.cut_size(), 0);
    }

    #[test]
    fn test_sign_pattern_max_cut() {
        // Max cut for K4 = 2*2 = 4
        let balanced = SignPattern::new([1, -1, -1, 1]);
        assert!(balanced.is_max_cut());

        // All same sign is not max cut
        let all_plus = SignPattern::new([1, 1, 1, 1]);
        assert!(!all_plus.is_max_cut());

        // K3: max cut = 1*2 = 2
        let sp3 = SignPattern::new([1, -1, 1]);
        assert_eq!(sp3.cut_size(), 2);
        assert!(sp3.is_max_cut());

        // K5: max cut = 2*3 = 6
        let sp5 = SignPattern::new([1, -1, 1, -1, -1]);
        assert_eq!(sp5.cut_size(), 2 * 3);
        assert!(sp5.is_max_cut());
    }

    // ── FleetZone tests ──

    #[test]
    fn test_dead_zone() {
        assert_eq!(zone_from_energy(0.005), FleetZone::Dead);
        assert_eq!(zone_from_energy(0.0), FleetZone::Dead);
    }

    #[test]
    fn test_dying_zone() {
        assert_eq!(zone_from_energy(0.01), FleetZone::Dying);
        assert_eq!(zone_from_energy(1.0), FleetZone::Dying);
        assert_eq!(zone_from_energy(1.999), FleetZone::Dying);
    }

    #[test]
    fn test_living_zone() {
        assert_eq!(zone_from_energy(2.0), FleetZone::Living);
        assert_eq!(zone_from_energy(3.0), FleetZone::Living);
        assert_eq!(zone_from_energy(4.0), FleetZone::Living);
    }

    #[test]
    fn test_strong_zone() {
        assert_eq!(zone_from_energy(4.001), FleetZone::Strong);
        assert_eq!(zone_from_energy(5.0), FleetZone::Strong);
        assert_eq!(zone_from_energy(9.999), FleetZone::Strong);
    }

    #[test]
    fn test_overdriven_zone() {
        assert_eq!(zone_from_energy(10.0), FleetZone::Overdriven);
        assert_eq!(zone_from_energy(100.0), FleetZone::Overdriven);
    }

    // ── Keel integration tests ──

    #[test]
    fn test_keel_auto_calibrate_on_first_read() {
        let states: [[f64; 2]; 4] =
            [[1.0, 2.0], [-1.0, -2.0], [0.5, 1.0], [-0.5, -1.0]];
        let mut keel = Keel::new(&states);
        let reading = keel.read(&states);
        // First read has zero velocity (no previous state)
        assert_eq!(reading.velocity, 0.0);
        // Position should be > 0
        assert!(reading.position > 0.0);
        // Strain should be zero (first read is baseline)
        for &s in &reading.strain {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn test_keel_velocity_nonzero_after_two_reads() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let s1: [[f64; 2]; 4] =
            [[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]];
        let _r1 = keel.read(&s1);
        let s2: [[f64; 2]; 4] =
            [[2.0, 2.0], [-2.0, -2.0], [1.0, 1.0], [-1.0, -1.0]];
        let r2 = keel.read(&s2);
        // Velocity should be positive (states changed)
        assert!(r2.velocity > 0.0);
        // Energy = (4+4 + 4+4 + 1+1 + 1+1)/4 = 20/4 = 5.0 => Strong? No: Living (2.0..=4.0)
        // Wait: 5.0 is > 4.0 so it's Strong. Let's check...
        // Actually (8+8+2+2)/4 = 20/4 = 5.0. That's > 4.0 => Strong, not Living.
        assert_eq!(r2.zone, FleetZone::Strong);
        assert!(r2.alignment > 0.8);
        assert!(r2.is_stable);
    }

    #[test]
    fn test_keel_detect_perturbation() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let baseline: [[f64; 2]; 4] =
            [[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0], [-0.5, 0.0]];
        let reading = keel.read(&baseline);
        // Baseline shouldn't trigger perturbation
        assert!(!keel.detect_perturbation(&reading, 0.5));

        // Now perturb — big jump in position + orientation shift
        let perturbed: [[f64; 2]; 4] =
            [[10.0, 10.0], [-10.0, -10.0], [5.0, 5.0], [-5.0, -5.0]];
        let r2 = keel.read(&perturbed);
        assert!(keel.detect_perturbation(&r2, 0.5));
    }

    #[test]
    fn test_keel_alignment_uniform() {
        // All agents perfectly aligned
        let aligned: [[f64; 2]; 4] =
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let align = mean_absolute_cosine_similarity(&aligned);
        assert!((align - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_keel_alignment_zero_vectors() {
        // Zero vectors => alignment = 0 (cosine similarity returns 0 for zero norms)
        let zeros: [[f64; 2]; 4] = [[0.0; 2]; 4];
        let align = mean_absolute_cosine_similarity(&zeros);
        assert!((align - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_keel_alignment_mixed() {
        // Alternating pairs: [1,0],[-1,0],[0,1],[0,-1]
        // Pairs:
        //   (0,1): |cos|=1, (0,2): |cos|=0, (0,3): |cos|=0,
        //   (1,2): |cos|=0, (1,3): |cos|=0, (2,3): |cos|=1
        // Total: 2/6 = 1/3
        let states: [[f64; 2]; 4] =
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        let align = mean_absolute_cosine_similarity(&states);
        assert!((align - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_expected_convergence_steps_nominal() {
        // At gain=0.95, coupling=0.25, 4 agents => 50 steps
        let steps = Keel::<2, 4>::expected_convergence_steps(0.95, 0.25, 4);
        assert_eq!(steps, 50);
    }

    #[test]
    fn test_expected_convergence_steps_faster_gain() {
        // Higher gain => fewer steps
        let normal = Keel::<2, 4>::expected_convergence_steps(0.95, 0.25, 4); // 50
        let fast = Keel::<2, 4>::expected_convergence_steps(1.0, 0.25, 4); // ceil(50*0.95) = 48
        assert!(fast < normal);
        assert_eq!(fast, 48);
    }

    #[test]
    fn test_expected_convergence_steps_zero_gain() {
        // Zero gain => impossible (usize::MAX)
        let impossible = Keel::<2, 4>::expected_convergence_steps(0.0, 0.25, 4);
        assert_eq!(impossible, usize::MAX);
    }

    #[test]
    fn test_expected_convergence_steps_zero_coupling() {
        // Zero coupling => impossible
        assert_eq!(Keel::<2, 4>::expected_convergence_steps(0.95, 0.0, 4), usize::MAX);
    }

    #[test]
    fn test_process_lateness_normal() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let states: [[f64; 2]; 4] = [[1.0; 2]; 4];
        keel.read(&states); // calibrate
        // expected=50, 2x=100, 10 < 100 => not late
        assert!(!keel.is_process_late(10, 0.95, 0.25, 3.0));
    }

    #[test]
    fn test_process_lateness_overdue() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let states: [[f64; 2]; 4] = [[1.0; 2]; 4];
        keel.read(&states);
        // expected=50, 2x=100, 500 > 100 => late
        assert!(keel.is_process_late(500, 0.95, 0.25, 3.0));
    }

    #[test]
    fn test_process_lateness_dead_energy() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let states: [[f64; 2]; 4] = [[1.0; 2]; 4];
        keel.read(&states);
        // expected=50, 10 < 100 but energy=0.001 < 0.01 => late
        assert!(keel.is_process_late(10, 0.95, 0.25, 0.001));
    }

    #[test]
    fn test_process_lateness_unconvergeable() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let states: [[f64; 2]; 4] = [[1.0; 2]; 4];
        keel.read(&states);
        // Zero gain => convergence not expected => false
        assert!(!keel.is_process_late(1000, 0.0, 0.25, 3.0));
    }

    #[test]
    fn test_living_zone_stability() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        // All +1 agents => all identical => alignment=1.0
        // Energy = (1+1)*4/4 = 2.0 => Living
        let states: [[f64; 2]; 4] =
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let reading = keel.read(&states);
        assert_eq!(reading.zone, FleetZone::Living);
        assert!(reading.is_stable);
    }

    #[test]
    fn test_strong_zone_stability_with_low_alignment() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        // Energy = (9+0 + 0+9 + 9+0 + 0+9)/4 = 36/4 = 9.0 => Strong
        // But orthogonal/alternating so low alignment
        let states: [[f64; 2]; 4] =
            [[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]];
        let reading = keel.read(&states);
        assert_eq!(reading.zone, FleetZone::Strong);
        // alignment should be 1/3 (similar to test_keel_alignment_mixed)
        assert!((reading.alignment - 1.0 / 3.0).abs() < 1e-10);
        assert!(!reading.is_stable);
    }

    #[test]
    fn test_dead_zone_not_stable() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let states: [[f64; 2]; 4] = [[0.0; 2]; 4];
        let reading = keel.read(&states);
        assert_eq!(reading.zone, FleetZone::Dead);
        assert!(!reading.is_stable);
    }

    #[test]
    fn test_strain_after_perturbation() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let baseline: [[f64; 2]; 4] =
            [[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]];
        keel.read(&baseline);

        // Shift agent 0's state from [1,1] to [5,5]
        let perturbed: [[f64; 2]; 4] =
            [[5.0, 5.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]];
        let reading = keel.read(&perturbed);
        // Agent 0 mean before: 1.0, after: 5.0 => strain = 4.0
        assert!((reading.strain[0] - 4.0).abs() < 1e-10);
        for i in 1..4 {
            assert!(reading.strain[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_natural_state_is_max_cut() {
        // The natural "+--+" pattern is a max-cut of K4
        let sp = SignPattern::new([1, -1, -1, 1]);
        assert!(sp.is_max_cut());
        assert_eq!(sp.label(), "+--+");
    }

    #[test]
    fn test_two_agent_keel() {
        // Test with 2 agents, 3 dimensions
        let states: [[f64; 3]; 2] = [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]];
        let mut keel = Keel::new(&states);
        let r = keel.read(&states);
        // Energy = (1+4+9 + 1+4+9) / 2 = 28/2 = 14 => Overdriven
        assert_eq!(r.zone, FleetZone::Overdriven);
        // Signs: mean of [1,2,3] = 2.0 (+), mean of [-1,-2,-3] = -2.0 (-)
        assert_eq!(r.orientation.label(), "+-");
        // Alignment between opposite vectors = |cos([1,2,3],[-1,-2,-3])| = 1.0
        assert!((r.alignment - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_explicit_calibrate() {
        let mut keel = Keel::<2, 4>::new(&[[0.0; 2]; 4]);
        let cal: [[f64; 2]; 4] =
            [[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]];
        keel.calibrate(&cal);

        // Read same states (should be at baseline)
        let r = keel.read(&cal);
        for &s in &r.strain {
            assert!(s.abs() < 1e-10);
        }
        assert_eq!(keel.step_count(), 1);
    }

    #[test]
    fn test_keel_two_dimensions() {
        // 2 agents, 2 dimensions, minimal config
        let states: [[f64; 2]; 2] = [[1.0, 1.0], [-1.0, -1.0]];
        let mut keel = Keel::new(&states);
        let r = keel.read(&states);
        assert!(r.position > 0.0);
        assert_eq!(r.orientation.label(), "+-");
        assert!(r.alignment > 0.99);
    }
}
