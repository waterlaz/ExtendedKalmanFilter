#include <random>
#include <iostream>

#include "../EKF.hpp"

#include <array>

using namespace ekf;
using namespace Eigen;

class SpeedMeasurement : public GenericMeasurementModel<float, 2, 1> {
public:
    static constexpr Real gatingProbability = 0.99;
    static std::pair<Vector<float, 1>, Matrix<float, 1, 2>> measure(const Vector<float, 2>& state) {
        Matrix<float, 1, 2> H; // Measurement matrix
        H << 0, 1; // We only measure speed
        return {H*state, H};
    }
    static std::array<size_t, 1> measurementAngleIndices() {
        return {0};
    }
};

class PositionMeasurement : public GenericMeasurementModel<float, 2, 1> {
public:
    static constexpr Real gatingProbability = 0.95;
    static std::pair<Vector<float, 1>, Matrix<float, 1, 2>> measure(const Vector<float, 2>& state) {
        Matrix<float, 1, 2> H; // Measurement matrix
        H << 1, 0; // We only measure position
        return {H*state, H};
    }
};

class Simple1DModel : public GenericProcessModel<float, 2> {
public:
    static std::tuple<State, StateJacobian, StateCovariance> predict(const State& state, float dt){
        float x = state[0]; // Position
        float v = state[1]; // Speed
        State x_pred(x + v*dt, v); // Simple linear motion
        StateJacobian F; // Jacobian of the state transition
        F << 1, dt,
             0, 1;
        StateCovariance Q;
        Q << 0.1, 0,
             0, 0.01; // Process noise covariance
        return {x_pred, F, 0.1*Q*dt};
    }
};

// EKF of a simple 1D system where the state is just a position and speed
class EKF_1D : public EKF<Simple1DModel, SpeedMeasurement, PositionMeasurement> {
public:
    EKF_1D() {
        this->initial_state<<0, 0; // Initial position and speed
        this->initial_state_covariance<<1, 0,
                                        0, 1; // Initial covariance
    }
    void addSpeed(float t, float speed) {
        Matrix<float, 1, 1> z; // Measurement
        z << speed; // We measure speed
        Matrix<float, 1, 1> R; // Measurement noise covariance
        R << 0.001; // Measurement noise
        this->addMeasurement<SpeedMeasurement>(t, z, R);
    }
    void addPosition(float t, float position) {
        Matrix<float, 1, 1> z; // Measurement
        z << position; // We measure position
        Matrix<float, 1, 1> R; // Measurement noise covariance
        R << 0.1; // Measurement noise
        this->addMeasurement<PositionMeasurement>(t, z, R);
    }
    float getPosition() {
        auto [state, covariance] = this->getLastState();
        return state[0]; // Return the estimated position
    }
};

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 0.1);
    EKF_1D ekf;
    EKF_1D ekf2;
    float t = 0.0;
    float dt = 0.1; // Time step
    for(int i=0; i<100; i++) {
        float x = sin(0.1*t); // True position
        float noisy_x = x + dis(gen); // Measurement noise
        ekf.addPosition(t, noisy_x); // Add noisy position measurement
        ekf2.addPosition(t, noisy_x); // Add noisy position measurement
        std::cout<<t<<" "<<x<<" "<<noisy_x<<" "<<ekf.getPosition()<<" "<<ekf2.getPosition()<<" "<<std::endl; // Print time, true position, and estimated position
        t += dt;
        float v = 0.1*cos(0.1*t); // True speed
        float noisy_v = v + 0.01*dis(gen); // Measurement noise
        ekf.addSpeed(t, noisy_v); // Add noisy speed measurement
        t += dt;
    }
}
