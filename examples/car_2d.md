# A 2D Car Extended Kalman Filter

Refer to the [Simple 1D](simple_1d.md) tutorial for a basic introduction to the EKF library. This tutorial covers more complex topics like:
* nonlinear process and measurement models,
* Mahalanobis gating for outlier rejection,
* angle-aware residuals for yaw,
* stateful measurement models


The filter estimates a 2D car state:

$$\vec{s} = (x,\ y,\ yaw,\ v,\ w,\ a)^T,$$

where:

- $x, y$ are position,
- $yaw$ is heading angle,
- $v$ is linear speed,
- $w$ is angular speed,
- $a$ is linear acceleration.

Three sensor types are fused:

1. linear and angular speed sensor ($v, w$),
2. pose sensor ($x, y, yaw$),
3. marker-distance sensor (distance to a known 2D marker position).

## Including the library

```cpp headers
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>

#include "EKF.hpp"

using namespace ekf;
using namespace Eigen;
```

## Defining the 2D process model

`Car2DModel` derives from `ProcessModelBase<double, 6>` and implements
a  nonlinear motion prediction and Jacobian $F$ for covariance propagation.

One of the key notable differences here is that the process model `Car2DModel` is
stateful and implements a *nonstatic* `predict(...)` method.
The `Car2DModel` class stores the process noise covariance `Q` as a member variable,
which can be set at runtime from configuration parameters.

Given the current car state with pose $(x, y, yaw)$, linear velocity $v$, angular velocity $w$ and linear acceleration $a$, the model predicts the next state after a time step $dt$ according to the following assumptions.
The car travels the distance $l = vdt + \frac{1}{2}adt^2$ in the average direction of the yaw angle $yaw + 0.5 w dt$, while the yaw angle changes from $yaw$ to $yaw + w dt$.
This advances $(x, y)$ to a new position
$(x + \cos(yaw + \frac{1}{2} w dt), \sin(yaw + \frac{1}{2} w dt))$.

Notice that during the update yaw is wrapped with `normalizeAngle(...)`
so it stays in the `[-pi, pi]` range.

```cpp process_model
// The Car 2D model is a simple kinematic model with 2d position, yaw angle,
// linear and angular velocity, and linear acceleration.
// The state vector is defined as [x, y, yaw, v, w, a].
class Car2DModel : public ProcessModelBase<double, 6> {
public:
    // process noise covariance
    StateCovariance Q; // this must be defined somewhere!!!
    // return the predicted state, the Jacobian, and the process noise covariance
    std::tuple<State, StateJacobian, StateCovariance>
    predict(const State& state, double dt) {
        // car position
        double x = state[0];
        double y = state[1];
        // car yaw angle
        double yaw = state[2];
        // car linear velocity
        double v = state[3];
        // car angular velocity
        double w = state[4];
        // car linear acceleration
        double a = state[5];
        // predicted traveled distance
        double l = v*dt + 0.5*a*dt*dt;
        // precompute the cos and sin of the yaw angle for the prediction step
        // note that we use the average yaw angle to improve accuracy
        double cos_ = cos(yaw + 0.5*w*dt);
        double sin_ = sin(yaw + 0.5*w*dt);
        State state_pred(
            x + l*cos_, // predict x position
            y + l*sin_, // predict y position
            // note the normalizeAngle to keep the yaw angle in the range [-pi, pi]
            normalizeAngle(yaw + w*dt), // predict yaw angle
            v + a*dt, // predict linear velocity
            w, // predict angular velocity
            a // predict linear acceleration
        );
        StateJacobian F;
        //   x  y      yaw        v              w                a
        F << 1, 0, -l*sin_, cos_*dt, -0.5*l*sin_*dt, 0.5*cos_*dt*dt, // x
             0, 1,  l*cos_, sin_*dt,  0.5*l*cos_*dt, 0.5*sin_*dt*dt, // y
             0, 0,       1,       0,             dt,              0, // yaw
             0, 0,       0,       1,             0,              dt, // v
             0, 0,       0,       0,             1,               0, // w
             0, 0,       0,       0,             0,               1; // a
        return {state_pred, F, Q*dt};
    }
};
```

### Configurable process covariance `Q`

`Q` is a model member (`ekf.process_model.Q`), so you can fill it from configuration values before running the filter.
The example fills it manually, but the same assignment can be made from parsed config parameters:

```cpp q_setup
ekf.process_model.Q <<
    q_x, 0, 0, 0, 0, 0,
    0, q_y, 0, 0, 0, 0,
    0, 0, q_yaw, 0, 0, 0,
    0, 0, 0, q_v, 0, 0,
    0, 0, 0, 0, q_w, 0,
    0, 0, 0, 0, 0, q_a;
```

This makes tuning process uncertainty easy without changing model code.

## Measurement model 1: speed (`v`, `w`)

`SpeedMeasurementModel` is linear and static:

```cpp speed_measurement
class SpeedMeasurementModel : public MeasurementModelBase<double, 6, 2> {
public:
    static std::pair<Vector<double, 2>, Matrix<double, 2, 6>>
    measure(const Vector<double, 6>& state) {
        ...
        return {H*state, H};
    }

    static constexpr Real gatingProbability = 0.95;
};
```

### Custom gating probability

Setting `gatingProbability` changes outlier rejection threshold for this sensor type.

- `0.95` means keep measurements that are statistically consistent at 95% confidence.
- Larger values are more permissive; smaller values reject more aggressive outliers.

This is configured per measurement model, so different sensors can have different gating behavior.

## Measurement model 2: pose (`x`, `y`, `yaw`)

`PositionMeasurementModel` is linear and also defines angle-aware innovation handling:

```cpp position_measurement
class PositionMeasurementModel : public MeasurementModelBase<double, 6, 3> {
public:
    static std::pair<Vector<double, 3>, Matrix<double, 3, 6>>
    measure(const Vector<double, 6>& state) {
        ...
        return {H*state, H};
    }

    static constexpr std::array<size_t, 1> measurementAngleIndices() {
        return {2};
    }
};
```

`measurementAngleIndices()` tells EKF that the third residual component is an angle and must be normalized.

## Measurement model 3: nonlinear, stateful marker distance

`MarkerDistanceMeasurementModel` demonstrates two advanced features:

1. **nonlinear measurement** (Euclidean distance to marker),
2. **stateful model instance** (holds `marker_position`).

Unlike the previous models, `measure` is **non-static** because it depends on per-instance data:

```cpp marker_distance_measurement
class MarkerDistanceMeasurementModel : public MeasurementModelBase<double, 6, 1> {
public:
    Vector<double, 2> marker_position;

    std::pair<Vector<double, 1>, Matrix<double, 1, 6>>
    measure(const Vector<double, 6>& state) const {
        ...
        return {Measurement(distance), H};
    }

    MarkerDistanceMeasurementModel(const Vector<double, 2>& marker_position)
        : marker_position(marker_position) {}
};
```

At update time, the EKF receives a model instance:

```cpp marker_add
this->addMeasurement(t, Vector<double, 1>{distance}, R_distance,
    MarkerDistanceMeasurementModel(marker_position));
```

This pattern is useful when each measurement needs custom context (for example different landmarks).

## Constructing the filter

```cpp filter
class EKF_Car2D : public EKF<
        Car2DModel,
        SpeedMeasurementModel,
        PositionMeasurementModel,
        MarkerDistanceMeasurementModel> {
public:
    ...
};
```

The filter stores per-sensor noise covariances (`R_speed`, `R_position`, `R_distance`) and exposes helpers to add each measurement type.

## Predicting state into the future

The example uses:

```cpp future_prediction
auto [state, covariance] = this->predictState(t_future);
```

to estimate the state at times where no measurement exists yet (for example `t + 0.5` in the loop).

This is useful for:

- controller look-ahead,
- latency compensation,
- generating trajectories at fixed output timestamps.

## Running the simulation

In `main()`:

1. Sensor noise levels are defined and mapped to `R_*` covariance matrices.
2. Process noise `Q` is set.
3. Synthetic ground truth trajectory is generated.
4. Measurements are added at different rates:
   - pose every 1.0 s,
   - marker distance every 0.4 s,
   - speed every 0.1 s.
5. Predicted future position is written to output files.

This shows that the EKF naturally handles multi-rate sensor fusion and prediction queries between updates.

## Summary

Compared to the 1D example, this 2D car tutorial adds:

1. A higher-dimensional nonlinear process model.
2. Per-model outlier gating via `gatingProbability`.
3. Angle-aware residual wrapping for yaw.
4. A **stateful, nonlinear, non-static** measurement model (`MarkerDistanceMeasurementModel`).
5. Runtime process-noise tuning through `ekf.process_model.Q` (easy to wire to config files).
6. Future-time state prediction using `predictState`.
