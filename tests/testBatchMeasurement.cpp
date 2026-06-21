#include <random>
#include <iostream>

#include "../EKF.hpp"

#include <array>

using namespace ekf;
using namespace Eigen;

class PositionMeasurement : public MeasurementModelBase<float, 2, 1> {
public:
    static std::pair<Vector<float, 1>, Matrix<float, 1, 2>> measure(const Vector<float, 2>& state) {
        Matrix<float, 1, 2> H; // Measurement matrix
        H << 1, 0; // We only measure position
        return {H*state, H};
    }
};

class Simple1DModel : public ProcessModelBase<float, 2> {
public:
    static std::tuple<State, StateJacobian, StateCovariance> predict(const State& state, float dt){
        float x = state[0]; // Position
        float v = state[1]; // Speed
        State x_pred(x + v*dt, v); // Simple linear motion
        //std::cout<<"prediction: "<<x<<" "<<v<<" "<<dt<<" "<<x_pred[0]<<" "<<x_pred[1]<<std::endl;
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
class EKF_1D : public EKF<Simple1DModel, PositionMeasurement> {
public:
    EKF_1D() {
        this->initial_state<<0, 0; // Initial position and speed
        this->initial_state_covariance<<1, 0,
                                        0, 1; // Initial covariance
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
    float getPositionAtTime(float t) {
        auto [state, covariance] = this->predictState(t);
        return state[0]; // Return the estimated position at time t
    }
    float getSpeed() {
        auto [state, covariance] = this->getLastState();
        return state[1]; // Return the estimated speed
    }
};

int main() {
    EKF_1D ekf;
    float last_time = 0.0;
    float dt = 0.1; // Time step
    float dx = 0.5; // True speed
    for(int i=0; i<100; i++) {
        float t = i * dt; // Time step
        float position = dx * t; // True position (for testing)
        ekf.addPosition(t, position);
        last_time = t;
    }
    float test_time = last_time + 10*dt;
    float test_position = dx * test_time; // True position at test time
    size_t numEntries = ekf.timeline.size();
    for(int i=0; i<100; i++) {
        float x = ekf.getPositionAtTime(test_time);
        if (std::abs(x - test_position) > 0.01) {
            std::cout << "Test failed at time " << test_time << ": expected " << test_position << ", got " << x << std::endl;
            return 1;
        }
        if(ekf.timeline.size() > numEntries + 1) {
            std::cout << "Test failed: timeline size increased after getPositionAtTime" << std::endl;
            return 1;
        }
    }
}
