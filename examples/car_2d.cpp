#include <random>
#include <iostream>
#include <fstream>
#include <cmath>

#include "EKF.hpp"

using namespace ekf;
using namespace Eigen;


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

class SpeedMeasurementModel : public MeasurementModelBase<double, 6, 2> {
public:
    static std::pair<Vector<double, 2>, Matrix<double, 2, 6>>
    measure(const Vector<double, 6>& state) {
        Matrix<double, 2, 6> H;
        H << 0, 0, 0, 1, 0, 0, // v
             0, 0, 0, 0, 1, 0; // w
        // In this linear example, the Jacobian equals H and the predicted measurement is H*state
        return {H*state, H};
    }
    // Accept measurements with 95% probability,
    // i.e. reject measurements that are too far away from the predicted measurement.
    static constexpr Real gatingProbability = 0.95;
};

class PositionMeasurementModel : public MeasurementModelBase<double, 6, 3> {
public:
    static std::pair<Vector<double, 3>, Matrix<double, 3, 6>>
    measure(const Vector<double, 6>& state) {
        Matrix<double, 3, 6> H;
        H << 1, 0, 0, 0, 0, 0, // x
             0, 1, 0, 0, 0, 0, // y
             0, 0, 1, 0, 0, 0; // yaw
        // In this linear example, the Jacobian equals H and the predicted measurement is H*state
        return {H*state, H};
    }
    // return the indices of the measurement vector that correspond to the angle.
    // 0 -- x
    // 1 -- y
    // 2 -- yaw is an angle and should be wrapped to [-pi, pi].
    static constexpr std::array<size_t, 1> measurementAngleIndices() {
        return {2};
    }
};

class MarkerDistanceMeasurementModel : public MeasurementModelBase<double, 6, 1> {
public:
    Vector<double, 2> marker_position; // position of the marker in the world frame
    // NOTE: the measure method is not static
    std::pair<Vector<double, 1>, Matrix<double, 1, 6>>
    measure(const Vector<double, 6>& state) const {
        Matrix<double, 2, 1> position = state.template head<2>();
        double distance = (position - marker_position).norm();
        double x = position[0];
        double y = position[1];
        double xm = marker_position[0];
        double ym = marker_position[1];
        Matrix<double, 1, 6> H;
        if(distance < 1e-6) {
            H.setZero();
        } else {
            H << (x - xm) / distance, (y - ym) / distance, 0, 0, 0, 0;
        }
        return {Measurement(distance), H};
    }
    MarkerDistanceMeasurementModel(const Vector<double, 2>& marker_position)
        : marker_position(marker_position) {}
};

class EKF_Car2D : public EKF<
        Car2DModel, // process model
        SpeedMeasurementModel, // first measurement model
        PositionMeasurementModel, // second measurement model
        MarkerDistanceMeasurementModel> { // third measurement model
public:
    Matrix<double, 2, 2> R_speed; // measurement noise covariance for speed
    Matrix<double, 3, 3> R_position; // measurement noise covariance for position
    Matrix<double, 1, 1> R_distance; // measurement noise covariance for distance
    EKF_Car2D() {
        // initialize the state and covariance
        this->initial_state << 0, 0, 0, 0, 0, 0;
        this->initial_state_covariance = StateCovariance::Identity() * 10.0;
    }
    // add a speed measurement to the filter at time t
    void addSpeed(double t, double speed_linear, double speed_angular) {
        Matrix<double, 2, 1> z;
        z << speed_linear, speed_angular;
        // add the measurement to the filter, specifying the measurement type as template parameter
        this->addMeasurement<SpeedMeasurementModel>(t, z, R_speed);
    }

    void addPosition(double t, const Vector<double, 3>& position) {
        this->addMeasurement<PositionMeasurementModel>(t, position, R_position);
    }

    void addDistanceToMarker(double t, double distance,
                             const Vector<double, 2>& marker_position)
    {
        this->addMeasurement(t, Vector<double, 1>{distance}, R_distance,
            MarkerDistanceMeasurementModel(marker_position));
    }

    Vector<double, 3> predictPosition(double t) {
        // query the estimated state from the filter and return the position component.
        // getLastState queries the filter state at the last measurement time.
        // In order to query the state at a specific time, use predictState(t).
        auto [state, covariance] = this->predictState(t);
        return state.template head<3>();
    }
};

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> dis(0.0, 1.0);


    double positionDeviation = 0.4;
    double angleDeviation = 0.01;
    double speedDeviation = 0.1;
    double angularSpeedDeviation = 0.01;
    double distanceDeviation = 0.01;

    EKF_Car2D ekf;
    ekf.R_position << positionDeviation*positionDeviation, 0, 0,
                      0, positionDeviation*positionDeviation, 0,
                      0, 0,       angleDeviation*angleDeviation;

    ekf.R_speed << speedDeviation*speedDeviation, 0,
                   0, angularSpeedDeviation*angularSpeedDeviation;

    ekf.R_distance << distanceDeviation*distanceDeviation;

    // don't forget to set the process noise covariance matrix Q in the process model!
    ekf.process_model.Q <<
        1e-4,    0,    0,    0,    0,    0, // x
           0, 1e-4,    0,    0,    0,    0, // y
           0,    0, 1e-5,    0,    0,    0, // yaw
           0,    0,    0, 1e-2,    0,    0, // v
           0,    0,    0,    0, 1e-2,    0, // w
           0,    0,    0,    0,    0, 1e-2; // a

    Vector<double, 2> marker_position;
    marker_position << 5.0, 5.0;

    double t = 0.0;
    double dt = 0.1;

    double yawPrev = std::nan("");

    std::ofstream trueFile("true_position");
    std::ofstream noisyFile("noisy_position");
    std::ofstream estimatedFile("estimated_position");

    for (int i = 0; i < 628; i++) {

        double x = 10.0 * sin(0.1*t);
        double y = 5.0 * sin(0.2*t);
        double dx = 10.0 * 0.1 * cos(0.1*t);
        double dy = 5.0 * 0.2 * cos(0.2*t);
        double v = sqrt(dx*dx + dy*dy);
        double yaw = atan2(dy, dx);

        if(i%10 == 0) {
            // add a position measurement every 1 second
            Vector<double, 3> position;
            position << x + dis(gen) * positionDeviation,
                        y + dis(gen) * positionDeviation,
                        yaw + dis(gen) * angleDeviation;
            ekf.addPosition(t, position);
            noisyFile<<position[0]<<" "<<position[1]<<std::endl;
        }

        if(i%4 == 0) {
            Vector<double, 2> position;
            position << x, y;
            double distance = (position - marker_position).norm();
            ekf.addDistanceToMarker(t,
                                    distance + dis(gen) * distanceDeviation,
                                    marker_position);
        }

        if(!std::isnan(yawPrev)) {
            double w = normalizeAngle(yaw - yawPrev) / dt;
            // add a speed measurement every 0.1 second
            ekf.addSpeed(t,
                         v + dis(gen) * speedDeviation,
                         w + dis(gen) * angularSpeedDeviation
            );
        }

        Vector<double, 3> position_est = ekf.predictPosition(t+0.5);

        trueFile<<x<<" "<<y<<std::endl;

        estimatedFile<<position_est[0]<<" "<<position_est[1]<<std::endl;

        t += dt;
        yawPrev = yaw;
    }
}
