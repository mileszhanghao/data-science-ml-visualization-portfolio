# Local Circuit Connectivity Modeling

Scientific-computing project modeling how neural connectivity structure affects impulse-response dynamics in a local circuit.

## Project Focus

The public version reframes a private notebook into reusable simulation code. It studies how observed connectivity, shuffled connectivity, connection probability, and reciprocal motifs affect the time required for neural activity to return to baseline after a stimulus.

## Methods

- Built connectivity matrices from synapse tables.
- Balanced excitatory and inhibitory inputs.
- Simulated firing-rate dynamics with an Euler update.
- Compared observed matrices against shuffled controls across random seeds.
- Measured recovery time after stimulus offset.
- Counted and removed reciprocal connections to test motif-level effects.

## Public Artifacts

- `local_circuit_simulation.py` contains the portfolio-safe simulation and analysis utilities.
- Raw neuroscience datasets and private notebook cells are not included.

## Skills Demonstrated

- NumPy simulation design.
- Pandas table filtering and matrix construction.
- Network connectivity analysis.
- Randomized controls and seed-based repeated experiments.
- Parameter sweeps and response-time measurement.
- Scientific visualization workflow design.

## Publication Note

The original notebook included course-specific prompts and scaffold. This directory keeps the reusable scientific-computing logic and a public summary of the analysis approach.
