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

Between measurements, the filter needs a way to predict how the system moves. This is the purpose of the process model, which describes how the state evolves over time.
This includes:
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


Process noise is defined by its covariance $Q$ and models small unmodelled accelerations that do not fit in the constant velocity assumption. Process noise reflects imperfections of the process model rather than sensor noise. A larger process noise covariance causes the filter to trust measurements more strongly and predictions less strongly.

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

The speed sensor observes only the speed $v$ component of the state $\vec{s}$.

The measurement function maps the state $(x, v)^T$ to the expected measurement (in this case, speed $v$). For this linear example, the function is represented by the matrix $H$:

$$H = [ 0 \\; 1 ].$$

Indeed, $v = H\cdot (x, v)^T$.

**NOTE:** The measurement model does not have to be linear. In case of a non-linear measurement model, $H$ is a Jacobian matrix that describes how the measurement changes with respect to the state.

Any measurement model must implement a *static* `measure` method that takes the current state as input and returns a pair of predicted measurement and Jacobian.
The method is static because measurement models are stateless in this implementation.
The filter does not store instances of them.
The predicted measurement is the expected value of the measurement given the current state, while the Jacobian describes how the measurement changes with respect to the state.

```cpp speed_measurement
// inherit from MeasurementModelBase and implement the static measure method
class SpeedMeasurement : public MeasurementModelBase<float, 2, 1> { //use float as scalar type, 2 as state dimension, and 1 as measurement dimension
public:
    // note that measurement is a vector of dimension 1, not a scalar
    static std::pair<Vector<float, 1>, Matrix<float, 1, 2>>
    measure(const Vector<float, 2>& state) {
        Matrix<float, 1, 2> H;
        H << 0, 1; // take only the speed component of the state as measurement
        // In this linear example, the Jacobian equals H and the predicted measurement is H*state
        return {H*state, H};
    }
};
```

## Defining a position measurement model

Similarly, the position sensor observes only the position, with measurement matrix

$$H = [ 1 \\; 0 ].$$

```cpp position_measurement
class PositionMeasurement : public MeasurementModelBase<float, 2, 1> { //use float as scalar type, 2 as state dimension, and 1 as measurement dimension
public:
    static std::pair<Vector<float, 1>, Matrix<float, 1, 2>>
    measure(const Vector<float, 2>& state) {
        Matrix<float, 1, 2> H;
        H << 1, 0; // take only the position component of the state as measurement
        return {H*state, H};
    }
};
```
Different sensors are represented by different measurement models.
The EKF automatically combines all measurements,
even if they arrive at different times or have different dimensions.

## Constructing the filter

To implement the filter, derive a class from `EKF` and specify the process model and measurement models as template parameters. The `EKF` class provides methods to add measurements and query the estimated state.


```cpp filter
class EKF_1D :
    public EKF<
        Simple1DModel, // process model
        SpeedMeasurement, // first measurement model
        PositionMeasurement> { // second measurement model
public:
    EKF_1D() {
        // initialize the state and covariance
        this->initial_state << 0, 0;
        this->initial_state_covariance << 1, 0,
                                          0, 1;
    }
    // add a speed measurement to the filter at time t
    void addSpeed(float t, float speed) {
        Matrix<float, 1, 1> z;
        z << speed;
        Matrix<float, 1, 1> R; // measurement noise covariance
        R << 0.001;
        // add the measurement to the filter, specifying the measurement type as template parameter
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
        // query the estimated state from the filter and return the position component.
        // getLastState queries the filter state at the last measurement time.
        // In order to query the state at a specific time, use predictState(t).
        auto [state, covariance] = this->getLastState();
        return state[0];
    }
};
```

## Simulating measurements

To demonstrate the filter, synthetic data is generated.

Define the true position to be $x(t) = \sin(0.1t)$.
The true speed is the derivative of the position, which is $v(t) = 0.1\cos(0.1t)$.
In a real application, measurements would come from sensors rather than from simulated sine waves.

Gaussian noise is added to both measurements before being passed to the filter. The position measurements are noisier than the speed measurements, which is reflected in the measurement noise covariance $R$.

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

1. Define the process model.
2. Define one or more measurement models.
3. Derive a filter class from `EKF`.
4. Initialize the state and covariance.
5. Add measurements as they arrive.
6. Query the estimated state.

This example demonstrates how multiple sensors can be fused using an Extended Kalman Filter.
Each sensor is described by its own measurement model,
while a common process model predicts the system between measurements.
The filter automatically combines all available information to produce a best estimate of the state.
