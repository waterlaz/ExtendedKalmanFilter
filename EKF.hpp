/* Copyright (c) 2026 Evgeniy Vodolazskiy (waterlaz)  */

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <tuple>


namespace EKFNamespace {
using namespace Eigen;
using std::function;
using std::vector;
using std::tuple;

template <typename Real>
Real normalizeAngle(Real angle) {
    constexpr Real PI = Real(3.14159265358979323846);
    return angle - 2.0 * PI * std::floor((angle + PI) / (2.0 * PI));
}

/*! A discrete point in time of a Kalman filter
 *  @tparam Real The floating point type to use (e.g. float, double)
 *  @tparam n The dimension of the state vector (can be set to Dynamic)
 * */
template <typename Real, int n>
class TimeStep {
public:
    using Mat = Matrix<Real, Dynamic, Dynamic>;
    using Vec = Matrix<Real, Dynamic, 1>;
    using State = Matrix<Real, n, 1>;
    using StateCovariance = Matrix<Real, n, n>;
    using StateTransition = Matrix<Real, n, n>;
    using Measurement = Matrix<Real, Dynamic, 1>;
    using MeasurementJacobian = Matrix<Real, Dynamic, n>;
    using MeasurementModel =
        function<tuple<Measurement, MeasurementJacobian>(const State&)>;
    /// @brief The time associated with the observation.
    Real time;
    /// @brief The state vector at the time of the observation (x).
    State state;
    /// @brief The covariance matrix of the state estimate (P).
    StateCovariance state_covariance;
    /// @brief The measurement vector associated with the observation (z).
    Measurement measurement;
    /**  @brief The function that maps the state to the tuple (measurement, measurement Jacobian).
     *
     *  This function takes a state vector as input and returns a tuple containing:
     *  - The expected measurement vector corresponding to the input state.
     *  - The measurement Jacobian matrix, which is the derivative of the measurement function with respect to the state.
     *
     *  For example, if the measurement is a non-linear function of the state, this function would compute both the expected measurement and its Jacobian for a given state.
     */
    MeasurementModel measurement_model;
    ///  @brief The covariance matrix of the measurement noise (R).
    Mat measurement_covariance;
    /**  @brief Indices of the angles in the state vector.
     *
     *  Any state variable that represents an angle (e.g. orientation)
     *  should have its index included in this vector.
     *  When an angle is updated, it will be wrapped to the range [-pi, pi].
     *  For example, if the state vector is [x, y, theta] and theta is an angle,
     *  then anglesIndices should be set to {2} (assuming 0-based indexing).
     */
    vector<int> angle_indices;

    /**
     * @brief Constructs an empty observation.
     *
     * Should be overwritten with actual time, state etc. before being used.
     */
    TimeStep() {}
    /**
     * @brief Constructs an observation with a given time,
     * optional state and no measurement.
     * @param time The time associated with the observation.
     */
    TimeStep(Real time,
                const State& state = undefined_state(),
                const StateCovariance& state_covariance = undefined_state_covariance())
        : time{time},
          state{state},
          state_covariance{state_covariance},
          measurement{no_measurement()},
          measurement_model{no_measurement_model()},
          measurement_covariance{no_measurement_covariance()} {}
    /**
     * @brief Constructs an observation with a given time, measurement,
     * measurement function and measurement noise covariance.
     * @tparam m The dimension of the measurement vector (can be set to Dynamic).
     * @param time The time associated with the observation.
     * @param measurement The actual measurement.
     * @param measurement_model The function that maps the state to the tuple
     * (measurement, measurement Jacobian).
     * @param measurement_covariance The covariance matrix of the measurement noise.
     * @param angle_indices The indices of the angles in the state vector.
     */
    template<int m>
    TimeStep(Real time,
             const Matrix<Real, m, 1>& measurement,
             const MeasurementModel& measurement_model,
             const Matrix<Real, m, m>& measurement_covariance,
             const vector<int>& angle_indices = {})
        : time{time},
          measurement{measurement},
          measurement_model{measurement_model},
          measurement_covariance{measurement_covariance},
          angle_indices{angle_indices} {}
    /**
     * @brief Constructs an observation with a given time, measurement,
     * measurement Jacobian and measurement noise covariance.
     *
     * It is a useful constructor when the measurement function is linear,
     * i.e. the measurement can be expressed as H * state,
     * where H is the measurement Jacobian.
     */
    template<int m>
    TimeStep(Real time,
             const Matrix<Real, m, 1>& measurement,
             const Matrix<Real, m, n>& measurement_jacobian,
             const Matrix<Real, m, m>& measurement_covariance,
             const vector<int>& angle_indices = {})
        : time{time},
          measurement{measurement},
          measurement_model{[measurement_jacobian](const State& state) {
              return std::make_tuple(measurement_jacobian * state,
                                     measurement_jacobian);
          }},
          measurement_covariance{measurement_covariance},
          angle_indices{angle_indices} {}

    /**
     * @brief The EKF prediction and update steps based on the previous TimeStep.
     * @param previous The previous TimeStep containing the prior state and covariance.
     * @param process_noise_covariance The covariance matrix of the process noise.
     * @param transition_jacobian The Jacobian matrix of the state transition function.
     * @param predicted_state The predicted state vector based on the state transition function.
     */
    void predict(const TimeStep<Real, n>& previous,
                 const StateCovariance& process_noise_covariance,
                 const StateTransition& transition_jacobian,
                 const State& predicted_state)
    {
        StateCovariance I = StateCovariance::Identity();
        auto& F = transition_jacobian;
        auto& Q = process_noise_covariance;
        auto P_prev = previous.state_covariance;
        auto& R = measurement_covariance;
        auto& x_pred = predicted_state;

        auto P_pred = F * sym(P_prev) * F.transpose() + Q;
        if(hasMeasurement()) {
            auto [z_pred, H] = measurement_model(x_pred);
            Vec y = measurement - z_pred;
            for(int i : angle_indices) {
                y[i] = normalizeAngle(y[i]);
            }

            Mat S = H * sym(P_pred) * H.transpose() + R;
            //Mat K = P_pred * H.transpose() * S.inverse(); but stable:
            Mat K = S.ldlt().solve(H * P_pred.transpose()).transpose();

            state = x_pred + K * y;
            // state_covariance = (I - K * H) * P_pred; but in Joseph form it is:
            state_covariance = (I - K * H) * sym(P_pred) * (I - K * H).transpose()
                             + K * R * K.transpose();
            state_covariance = sym(state_covariance);
        } else {
            state = x_pred;
            state_covariance = P_pred;
        }
    }

    /**  @brief Checks if the observation has a measurement associated with it */
    ///
    ///  If the observation has no measurement,
    ///  it represents a discrete point in time used for a prediction step only.
    ///  In order to accurately predict the state in non-linear systems,
    ///  these timesteps are necessary to perform intermediate prediction steps.
    ///  @return true if the measurement vector is non-empty, false otherwise
    ///
    bool hasMeasurement() const {
        return measurement.size() > 0;
    }
private:
    static StateCovariance undefined_state_covariance() {
        return StateCovariance::Constant(std::numeric_limits<Real>::quiet_NaN());
    }
    static State undefined_state() {
        return State::Constant(std::numeric_limits<Real>::quiet_NaN());
    }
    static Measurement no_measurement() {
        return Matrix<Real, 0, 1>();
    }
    static Mat no_measurement_covariance() {
        return Matrix<Real, 0, 0>();
    }
    static MeasurementJacobian no_measurement_jacobian() {
        return Matrix<Real, 0, n>();
    }
    static MeasurementModel no_measurement_model() {
        return [](const State&) {
            return std::make_tuple(no_measurement(), no_measurement_jacobian());
        };
    }
    template <typename Derived>
    static inline auto sym(const Eigen::MatrixBase<Derived>& m) {
        return m.template selfadjointView<Eigen::Lower>();
    }
};

} // namespace EKFNamespace
