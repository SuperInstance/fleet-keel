# fleet-keel ⚒️

**A 5-dimensional self-orientation system for fleet agents using only internal dynamics.**

The keel provides a fleet agent with process-relative orientation — no external clock, no timestamps, no shared synchronization primitives. Just internal state dynamics.

## Theory

In a multi-agent system where agents coordinate through coupled oscillators, the keel acts as a **stabilizing keel** (like a ship's keel): it gives each agent a sense of where it is in the collective manifold without requiring external reference frames.

The keel measures 5 dimensions from the raw agent state vectors:

| # | Dimension | Definition | What It Tells You |
|---|-----------|-----------|-------------------|
| 1 | **Position** | Σ\|\|stateᵢ\|\|² / N | Energy — where the fleet is on the manifold |
| 2 | **Orientation** | sign(meanᵢ) for each agent | Mood — 16 possible sign patterns for 4 agents |
| 3 | **Velocity** | \|\|state(t) − state(t−1)\|\|² | Rate of state change |
| 4 | **Strain** | \|meanᵢ(t) − baseline_meanᵢ\| | Deviation from attractor baseline |
| 5 | **Alignment** | mean \|cos_sim(agentᵢ, agentⱼ)\| | Coherence of the fleet |

### Zones (from 53 GPU experiments)

| Zone | Energy Range | Correlation | Meaning |
|------|-------------|-------------|---------|
| Dead | < 0.01 | — | System collapsed to origin |
| Dying | 0.01–2.0 | — | Weak, barely alive |
| Living | 2.0–4.0 | > 0.85 | Healthy coherent state |
| Strong | 4.0–10.0 | > 0.9 | Strong coherent state |
| Overdriven | ≥ 10.0 | — | May become unstable |

### Verified Constants

- **Melting point:** coupling ≈ 0.67 / N^1.06 (above this, orientation diversity collapses)
- **Gain edge:** gain > 0.85 required for convergence
- **Natural state:** `+--+` (max-cut of K₄ coupling graph)

## Usage

```rust
use fleet_keel::{Keel, KeelReading, SignPattern, FleetZone};

// 4 agents, 2-dimensional states
let states = [
    [1.0, 2.0],
    [-1.0, -2.0],
    [0.5, 1.0],
    [-0.5, -1.0],
];

let mut keel = Keel::new(&states);
let reading: KeelReading<4> = keel.read(&states);

println!("Energy: {}", reading.position);        // Position
println!("Mood: {}", reading.orientation);        // e.g. "+--+"
println!("Velocity: {}", reading.velocity);       // 0.0 on first read
println!("Strain: {:?}", reading.strain);         // ~0 on first read
println!("Alignment: {}", reading.alignment);
println!("Zone: {:?}", reading.zone);
println!("Stable: {}", reading.is_stable);

// Is a peer agent late?
if keel.is_process_late(120, 0.95, 0.25, 1.5) {
    println!("Other agent is behind schedule");
}
```

### Process-Relative Timing

The keel replaces wall-clock time with **process-relative** timing:

```rust
// How many steps should peer take at gain=0.95, coupling=0.25?
let expected = Keel::<2, 4>::expected_convergence_steps(0.95, 0.25, 4);
assert_eq!(expected, 50);

// Is that peer late after 500 steps with dead energy?
if keel.is_process_late(500, 0.95, 0.25, 0.0001) {
    // Peer has likely crashed or diverged
}
```

### Perturbation Detection

```rust
if keel.detect_perturbation(&reading, 0.5) {
    // Something disturbed the fleet — reevaluate
}
```

### Sign Patterns

```rust
let sp = SignPattern::new([1, -1, -1, 1]);
assert_eq!(sp.label(), "+--+");
assert_eq!(sp.cut_size(), 4);  // edges across the partition
assert!(sp.is_max_cut());      // balanced partition of K₄
```

## Research

This crate is the Rust implementation of the keel system verified in 53 GPU experiments. For the full research, see the [Casting Call knowledge base](https://github.com/SuperInstance/casting-call) and related fleet repos.

## License

MIT
