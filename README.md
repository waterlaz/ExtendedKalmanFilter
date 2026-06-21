# ExtendedKalmanFilter

A header-only C++20 (also supports older version) Extended Kalman Filter (EKF) library based on Eigen.

This library is designed for real-time sensor fusion with:
- Multiple measurement model types
- Out-of-order timestamp insertion with automatic re-propagation
- Numerically stable covariance updates
- Optional angle normalization and statistical gating for outlier rejection

The public API is in [`EKF.hpp`](./EKF.hpp).

## Features

- **Header-only**: include `EKF.hpp` and start using it.
- **Strongly typed models**: process and measurement interfaces are checked with C++20 concepts with fallbacks for older compilers.
- **Multiple measurement models**: use one EKF instance with different sensor types.
- **Predictable performance**: no dynamic memory allocations after initialization.
- **Out-of-order updates**: insert old measurements; subsequent states are recomputed automatically.
- **Prediction-only steps**: query/filter at arbitrary times.
- **Numerical stability helpers**:
  - Joseph-form covariance update
  - Symmetry-preserving covariance handling
  - Positive-definite checks and recovery path
- **Built-in outlier gating**: Mahalanobis-distance threshold from configurable `gatingProbability`.
- **Angle-aware innovations**: normalize selected measurement residual components to `[-pi, pi]`.

## Requirements

- C++11 compiler (or C++20 compiler for concepts support)
- [Eigen](https://eigen.tuxfamily.org/) (`Eigen/Dense`)

## Project layout

- `EKF.hpp` – library implementation and API documentation (Doxygen style comments)
- `examples/simple_1d.cpp` – minimal runnable usage example
- `tests/testTimeLine.cpp` – timeline ordering test executable
- `tests/testBatchMeasurement.cpp` – tests not flooding the timeline when querying states at the same time. Also tests for simple prediction being sane.
- `Doxyfile` – Doxygen configuration

## Quick start

1. Add `EKF.hpp` to your include path.
2. Define:
   - a process model by inheriting `ekf::GenericProcessModel<Real, n>`
   - one or more measurement models by inheriting `ekf::GenericMeasurementModel<Real, n, m>`
3. Instantiate `ekf::EKF<ProcessModel, MeasurementModel1, ...>`.
4. Set `initial_state` and `initial_state_covariance`.
5. Feed measurements with `addMeasurement<Model>(time, z, R)` and/or request predicted states with `getState(time)`.

## Build test
...

## Core API reference

Main classes and templates in `EKF.hpp`:

- `ekf::GenericProcessModel<Real, n>`
- `ekf::GenericMeasurementModel<Real, n, m>`
- `ekf::MeasurementStep<Model>`
- `ekf::EKF<ProcessModel, MeasurementModels...>`

Important EKF methods:

- `addMeasurement<Model>(time, measurement, measurement_covariance)`
- `addNoMeasurement(time)`
- `getState(time)`
- `getLastState()`
- `reset()`

## Documentation generation

Generate API docs with Doxygen:

```bash
doxygen Doxyfile
```

## License
BSD 3-Clause License (see [`LICENSE`](./LICENSE)).
