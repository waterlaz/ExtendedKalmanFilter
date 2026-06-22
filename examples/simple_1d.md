# A Simple 1D Extended Kalman Filter

In this example we estimate the position and speed of an object moving in one dimension. Two independent sensors are available:

* a position sensor,
* a speed sensor.

The state vector $\vec{s}$ consists of the position $x$ and speed $v$:

$$\vec{s} = (x, v)^T.$$

To use the Kalman filter one has to provide the following components:
* a [process model](#defining-the-process-model) that describes how the state evolves over time,
* a measurement model for each sensor that describes how the measurements relate to the state:
    * [a speed measurement model](#defining-a-speed-measurement-model),
    * [a position measurement model](#defining-a-position-measurement-model).

## Including the library

The filter implementation uses Eigen for vectors and matrices. All you need to start using the Kalman filter is include the `EKF.hpp` header file. To run the simulation we also include `random` and `iostream`, which are not mandatory for the filter itself.

```cpp headers
#include <random>
#include <iostream>

#include "EKF.hpp"

using namespace ekf;
using namespace Eigen;
```

## Defining the process model

The process model describes how the state evolves over time. This includes:
* the state transition function, which predicts the next state $(x_{k+1}, v_{k+1})$ based on the current state $(x_k, v_k)$ and time interval $dt$,
* the Jacobian $F$ of the state transition function, which is used to propagate the uncertainty of the state,
* the process noise covariance $Q$, which defines how uncertainty grows over time.

This example assumes constant velocity motion.

During a time interval $dt$ the position changes proportional to the velocity:

$$ x_{k+1} = x_k + v_k dt, $$

while the speed remains the same,

$$v_{k+1}=v_k.$$

The Jacobian $F$ of the state transition function is

$$ F =
\begin{bmatrix}
\frac{\partial x_{k+1}}{\partial x_k} & \frac{\partial x_{k+1}}{\partial v_k}\\\
\frac{\partial v_{k+1}}{\partial x_k} & \frac{\partial v_{k+1}}{\partial v_k}
\end{bmatrix} =
\begin{bmatrix}
1 & dt\\\
0 & 1
\end{bmatrix}.$$


Process noise is defined by its covariance $Q$ and models small unmodelled accelerations that do not fit in the constant velocity assumption.
The covariance of the process noise is typically assumed to be proportional to the time interval $dt$: $Q\cdot dt$. For a simple model, we assume diagonal process noise covariance, where $Q_{11}$ models the uncertainty in position and $Q_{22}$ models the uncertainty in speed.

To define the process model, derive a class from `ProcessModelBase` and implement the `predict` method, which returns the predicted state, the Jacobian, and the process noise covariance. The `EKF` class keeps an instance of the process model class and calls the `predict` method whenever a new measurement is added.

```cpp process_model
// inherit from ProcessModelBase for convenience and implement the predict method
class Simple1DModel : public ProcessModelBase<float, 2> { //use float as scalar type and 2 as state dimension
public:
    // return the predicted state, the Jacobian, and the process noise covariance
    std::tuple<State, StateJacobian, StateCovariance>
    predict(const State& state, float dt) {

        float x = state[0]; // position
        float v = state[1]; // speed

        State state_pred(x + v*dt, v); // predict next state

        StateJacobian F;
        F << 1, dt, // position depends on both position and speed
             0, 1; // speed depends only on speed

        StateCovariance Q;
        Q << 0.01, 0, // process noise for position
             0, 0.01; // process noise for speed

        return {state_pred, F, Q*dt};
    }
};
```

## Defining a speed measurement model

The speed sensor observes only the speed component of the state,

[
z = v.
]

The corresponding measurement matrix is

[
H =
\begin{bmatrix}
0 & 1
\end{bmatrix}.
]

The measurement model returns both the predicted measurement and the Jacobian.

```cpp speed_measurement
class SpeedMeasurement : public MeasurementModelBase<float, 2, 1> {
public:
    static std::pair<Vector<float, 1>, Matrix<float, 1, 2>>
    measure(const Vector<float, 2>& state) {
        Matrix<float, 1, 2> H;
        H << 0, 1;
        return {H*state, H};
    }
};
```

## Defining a position measurement model

Similarly, the position sensor observes only the position,

[
z = p,
]

with measurement matrix

[
H =
\begin{bmatrix}
1 & 0
\end{bmatrix}.
]

```cpp position_measurement
class PositionMeasurement : public MeasurementModelBase<float, 2, 1> {
public:
    static std::pair<Vector<float, 1>, Matrix<float, 1, 2>>
    measure(const Vector<float, 2>& state) {
        Matrix<float, 1, 2> H;
        H << 1, 0;
        return {H*state, H};
    }
};
```


## Constructing the filter

The filter combines the process model with both measurement models.

Initially, we assume that the object is located at the origin and is stationary.

```cpp filter
class EKF_1D :
    public EKF<
        Simple1DModel,
        SpeedMeasurement,
        PositionMeasurement> {

public:
    EKF_1D() {

        this->initial_state << 0, 0;

        this->initial_state_covariance
            << 1, 0,
               0, 1;
    }

    void addSpeed(float t, float speed) {

        Matrix<float,1,1> z;
        z << speed;

        Matrix<float,1,1> R;
        R << 0.001;

        this->addMeasurement<SpeedMeasurement>(t, z, R);
    }

    void addPosition(float t, float position) {

        Matrix<float,1,1> z;
        z << position;

        Matrix<float,1,1> R;
        R << 0.1;

        this->addMeasurement<PositionMeasurement>(t, z, R);
    }

    float getPosition() {

        auto [state, covariance] = this->getLastState();

        return state[0];
    }
};
```

## Simulating measurements

To demonstrate the filter, we generate synthetic data.

The true position is

[
p(t)=\sin(0.1t),
]

and the true speed is

[
v(t)=0.1\cos(0.1t).
]

Gaussian noise is added to both measurements.

```cpp main
int main() {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<float> dis(0.0, 0.1);

    EKF_1D ekf;

    float t = 0.0;
    float dt = 0.1;

    for (int i = 0; i < 100; i++) {

        float x = sin(0.1*t);
        float noisy_x = x + dis(gen);

        ekf.addPosition(t, noisy_x);

        float v = 0.1*cos(0.1*t);
        float noisy_v = v + 0.01*dis(gen);

        ekf.addSpeed(t, noisy_v);

        std::cout
            << t << " "
            << x << " "
            << noisy_x << " "
            << ekf.getPosition()
            << std::endl;

        t += dt;
    }
}
```

## Summary

This example demonstrates the basic structure of an Extended Kalman Filter:

1. Define one or more measurement models.
2. Define the process model.
3. Derive a filter class from `EKF`.
4. Initialize the state and covariance.
5. Add measurements as they arrive.
6. Query the estimated state.

Despite its simplicity, this example already illustrates one of the strengths of the library: multiple measurement types can be fused transparently by a single filter.
