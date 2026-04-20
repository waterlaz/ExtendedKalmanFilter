/* Copyright (c) 2026 Evgeniy Vodolazskiy (waterlaz)  */

#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <functional>
#include <map>
#include <tuple>
#include <vector>

namespace EKFNamespace {
using namespace Eigen;
using std::function;
using std::vector;
using std::tuple;

/** @brief Normalizes an angle to the range [-pi, pi].
 *
 *  This function takes an angle in radians and wraps it to the range [-pi, pi].
 *  It is useful for ensuring that angle measurements and state variables representing angles remain within a consistent range,
 *  which can help prevent issues with angle discontinuities in the EKF update step.
 *
 *  @tparam Real The floating point type of the angle (e.g. float, double).
 *  @param angle The angle to normalize, in radians.
 *  @return The normalized angle in the range [-pi, pi].
 */
template <typename Real>
Real normalizeAngle(Real angle) {
    constexpr Real PI = Real(3.14159265358979323846);
    constexpr Real TWO_PI = Real(2) * PI;
    return angle - TWO_PI * std::floor((angle + PI) / TWO_PI);
}

template <typename Real, int n>
bool isMatrixPositiveDefinite(const Matrix<Real, n, n>& A){
    if(A.rows() != A.cols() || !A.isApprox(A.transpose())){
        return false;
    }
    LDLT<Matrix<Real, n, n>> ldlt(A);
    return ldlt.info() == Success && ldlt.isPositive();
}

/*! A discrete point in time of a Kalman filter.
 *  Most of the filter math is hidden within this class in the predict() method.
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
    using MeasurementModel = function<tuple<Measurement, MeasurementJacobian>(const State&)>;
    /// @brief The state vector at the time of the observation (x).
    State state;
    /// @brief The covariance matrix of the state estimate (P).
    StateCovariance state_covariance;
    /// @brief The measurement vector associated with the observation (z).
    Measurement measurement;
    /**  @brief The function that maps the state to the tuple (measurement, Jacobian).
     *
     *  This function takes a state vector as input and returns a tuple containing:
     *  - The expected measurement vector corresponding to the input state.
     *  - The measurement Jacobian matrix,
     *  which is the derivative of the measurement function with respect to the state.
     *
     *  This is useful for non-linear measurements.
     */
    MeasurementModel measurement_model;
    ///  @brief The covariance matrix of the measurement noise (R).
    Mat measurement_covariance;
    /**  @brief Indices of the angles in the measurement vector.
     *
     *  Any measurement variable that represents an angle (e.g. orientation)
     *  should have its index included in this vector.
     *  When an angle is updated, it will be wrapped to the range [-pi, pi].
     *  For example, if the state vector is [x, y, theta] and theta is an angle,
     *  then anglesIndices should be set to {2} (assuming 0-based indexing).
     */
    vector<int> measurement_angle_indices;
    /**
     * @brief Constructs an observation with a given time,
     * optional state and no measurement.
     * @param time The time associated with the observation.
     */
    TimeStep(const State& state = undefined_state(),
             const StateCovariance& state_covariance = undefined_state_covariance())
        : state{state},
          state_covariance{state_covariance},
          measurement{no_measurement()},
          measurement_model{no_measurement_model()},
          measurement_covariance{no_measurement_covariance()} {}
    /**
     * @brief Constructs an observation with a given measurement,
     * measurement function and measurement noise covariance.
     * @tparam m The dimension of the measurement vector (can be set to Dynamic).
     * @param measurement The actual measurement.
     * @param measurement_model The function that maps the state to the tuple
     * (measurement, measurement Jacobian).
     * @param measurement_covariance The covariance matrix of the measurement noise.
     * @param angle_indices The indices of the angles in the state vector.
     */
    template<int m>
    TimeStep(const Matrix<Real, m, 1>& measurement,
             const MeasurementModel& measurement_model,
             const Matrix<Real, m, m>& measurement_covariance,
             const vector<int>& angle_indices = {})
        : measurement{measurement},
          measurement_model{measurement_model},
          measurement_covariance{measurement_covariance},
          measurement_angle_indices{angle_indices}
    {
        assert(isMatrixPositiveDefinite(measurement_covariance));
        assert(measurement.size() == measurement_covariance.rows());
    }
    /**
     * @brief Constructs an observation with a given measurement,
     * measurement Jacobian and measurement noise covariance.
     *
     * It is a useful constructor when the measurement function is linear,
     * i.e. the measurement can be expressed as H * state,
     * where H is the measurement Jacobian.
     */
    template<int m>
    TimeStep(const Matrix<Real, m, 1>& measurement,
             const Matrix<Real, m, n>& measurement_jacobian,
             const Matrix<Real, m, m>& measurement_covariance,
             const vector<int>& measurement_angle_indices = {})
        : measurement{measurement},
          measurement_model{[measurement_jacobian](const State& state) {
              return std::make_tuple(measurement_jacobian * state,
                                     measurement_jacobian);
          }},
          measurement_covariance{measurement_covariance},
          measurement_angle_indices{measurement_angle_indices}
    {
        assert(isMatrixPositiveDefinite(measurement_covariance));
        assert(measurement.size() == measurement_covariance.rows());
        assert(measurement_jacobian.rows() == measurement.size());
    }

    /**
     * @brief The EKF prediction and update steps based on the previous TimeStep.
     * @param previous The previous TimeStep containing the prior state and covariance.
     * @param process_noise_covariance The covariance matrix of the process noise.
     * @param transition_jacobian The Jacobian matrix of the state transition function.
     * @param predicted_state The predicted state vector based on the state transition function.
     */
    void update(const TimeStep<Real, n>& previous,
                const State& predicted_state,
                const StateTransition& transition_jacobian,
                const StateCovariance& process_noise_covariance)
    {
        StateCovariance I = StateCovariance::Identity();
        const auto& F = transition_jacobian;
        const auto& Q = process_noise_covariance;
        const auto& P_prev = previous.state_covariance;
        const auto& R = measurement_covariance;
        const auto& x_pred = predicted_state;

        assert(previous.hasEstimatedState());
        assert(isMatrixPositiveDefinite(Q));
        assert(isMatrixPositiveDefinite(R));
        assert(isMatrixPositiveDefinite(P_prev));

        auto P_pred = F * sym(P_prev) * F.transpose() + Q;
        //std::cout<<"P_pred = \n"
        //         <<F * sym(P_prev) * F.transpose()<<"\n+\n"<<Q<<"\n\n\n";
        if(hasMeasurement()) {
            auto [z_pred, H] = measurement_model(x_pred);
            assert(z_pred.size() == measurement.size());
            assert(H.rows() == measurement.size());
            Vec y = measurement - z_pred;
            for(int i : measurement_angle_indices) {
                y[i] = normalizeAngle(y[i]);
            }

            Mat S = H * sym(P_pred) * H.transpose() + R;
            //Mat K = P_pred * H.transpose() * S.inverse(); but stable:
            Mat K = S.ldlt().solve(H * P_pred).transpose();

            state = x_pred + K * y;
            // state_covariance = (I - K * H) * P_pred; but in Joseph form:
            state_covariance = (I - K * H) * sym(P_pred) * (I - K * H).transpose()
                             + K * R * K.transpose();
            state_covariance = sym(state_covariance);
        } else {
            state = x_pred;
            state_covariance = P_pred;
        }
    }

    /**  @brief Checks if the TimeStep has a measurement associated with it
     *
     *  If the TimeStep has no measurement,
     *  it represents a discrete point in time used for a prediction step only.
     *  In order to accurately predict the state in non-linear systems,
     *  these timesteps are necessary to perform intermediate prediction steps.
     *  @return true if the measurement vector is non-empty, false otherwise
     */
    bool hasMeasurement() const {
        return measurement.size() > 0 && measurement_covariance.size() > 0;
    }
    /** @brief Checks if the TimeStep has an estimated state.
     *
     *  A TimeStep has an estimated state if its state vector and covariance matrix
     *  do not contain NaN values. A TimeStep without an estimated state
     *  cannot be used as a previous TimeStep for the update() method.
     */
    bool hasEstimatedState() const {
        return !state.hasNaN() && !state_covariance.hasNaN();
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
    // a usefull optimisation to treat symmetric matrices in computations
    // also improves numerical stability.
    template <typename Derived>
    static inline auto sym(const Eigen::MatrixBase<Derived>& m) {
        return m.template selfadjointView<Eigen::Lower>();
    }
};

template <typename Real, int n>
class EKF {
public:
    using StateCovariance = Matrix<Real, n, n>;
    using StateJacobian = Matrix<Real, n, n>;
    using State = Matrix<Real, n, 1>;
    using MeasurementModel = typename TimeStep<Real, n>::MeasurementModel;
    /// @brief maximum number of TimeSteps to keep in the TimeLine (or 0 to ingore).
    size_t max_history_count = 1000;
    /// @brief maximum period of time to keep in the TimeLine (or negative to ignore).
    Real max_history_time = -1;
    /// @brief stores TimeSteps sorted by time.
    std::multimap<Real, TimeStep<Real, n>> timeline;
    /// @brief the initial state vector for the EKF when there are no previous steps.
    State initial_state;
    /// @brief the initial state covariance matrix for the EKF.
    StateCovariance initial_state_covariance;
    /// @brief resets the EKF to the initial state.
    void reset(Real t, const State& state) {
        timeline.clear();
    }
    /**  @brief The function that maps the previous state and time duration to
     * the tuple (predicted state, Jacobian, state transition noise covariance).
     *
     *  This function should take a state vector and passed time as input
     *  and return a triplet containing:
     *  - The predicted new state vector after the time dt.
     *  - The state Jacobian matrix, how the next state depends on the previous state.
     *  - The covariance matrix of the process noise for the state transition.
     *  In many cases, the covariance matrix can be Q*dt for some constant Q.
     */
    virtual std::tuple<State, StateJacobian, StateCovariance> predict(
        const State& state, Real dt) = 0;
    /** @brief Adds a new TimeStep and performs the EKF prediction and update steps.
     *
     *  The EKF prediction and update steps will be performed for the new
     *  TimeStep and all subsequent TimeSteps in the TimeLine.
     *  @param time The time associated with the new TimeStep to add to the TimeLine.
     *  @param timestep The TimeStep to add to the TimeLine and perform EKF steps for.
     *  @return reference to the added TimeStep in the TimeLine.
     */
    const TimeStep<Real, n>& addTimeStep(Real time,
                                         const TimeStep<Real, n>& timestep)
    {
        if(timeline.empty()) {
            timeline.insert({time,
                TimeStep<Real, n>(initial_state, initial_state_covariance)});
        } else if(timeline.begin()->first > time) {
            return timeline.begin()->second;
        }
        auto res = timeline.insert({time, timestep});
        for(auto cur = res; cur != timeline.end(); cur++) {
            if(cur != timeline.begin()) {
                auto prev = std::prev(cur);
                Real dt = cur->first - prev->first;
                auto [x_pred, F, Q] = predict(prev->second.state, dt);
                cur->second.update(prev->second, x_pred, F, Q);
            }
        }
        // remove old timesteps if we exceed the limits
        while( timeline.begin() != res && // explicitly forbid removing the new item
               std::next(timeline.begin()) != res && // and the one before it
               ((max_history_count>0 && timeline.size() > max_history_count) ||
                (max_history_time>0 && totalTime() > max_history_time)) ) {
            timeline.erase(timeline.begin());
        }
        return res->second;
    }

    /// @brief Adds a new TimeStep with no measurement at the given time and
    /// performs the EKF prediction step based on the previous TimeStep.
    const TimeStep<Real, n>& addNoMeasurement(Real time) {
        return addTimeStep(time, TimeStep<Real, n>());
    }
    /** @brief Adds a new general TimeStep with a measurement at the given time.
     *
     * @tparam m The dimension of the measurement vector (can be set to Dynamic).
     * @param time The time associated with the measurement.
     * @param measurement The actual measurement vector.
     * @param measurement_model The function that maps the state to the tuple
     * (measurement, measurement Jacobian).
     * @param measurement_covariance The covariance matrix of the measurement noise.
     * @param measurement_angle_indices The indices of angles in the measurement vec.
     * @return reference to the added (or joined) TimeStep in the TimeLine.
     */
    template<int m>
    const TimeStep<Real, n>& addMeasurement(
                             Real time,
                             const Matrix<Real, m, 1>& measurement,
                             const MeasurementModel& measurement_model,
                             const Matrix<Real, m, m>& measurement_covariance,
                             const vector<int>& measurement_angle_indices = {})
    {
        return addTimeStep(time,
            TimeStep<Real, n>(measurement, measurement_model,
            measurement_covariance, measurement_angle_indices));
    }

    /** @brief Adds a new TimeStep with a linear measurement at the given time.
     *
     *  This is a useful overload of the addMeasurement() method
     *  for the case of linear measurements, where the measurement can be expressed
     *  as H * state, where H is the measurement Jacobian.
     *
     * @tparam m The dimension of the measurement vector (can be set to Dynamic).
     * @param time The time associated with the measurement.
     * @param measurement The actual measurement vector.
     * @param measurement_jacobian The Jacobian matrix of the linear measurement.
     * @param measurement_covariance The covariance matrix of the measurement noise.
     * @param measurement_angle_indices The indices of angles in the measurement vec.
     * @return reference to the added (or joined) TimeStep in the TimeLine.
     */
    template<int m>
    const TimeStep<Real, n>& addMeasurement(
                             Real time,
                             const Matrix<Real, m, 1>& measurement,
                             const Matrix<Real, m, n>& measurement_jacobian,
                             const Matrix<Real, m, m>& measurement_covariance,
                             const vector<int>& measurement_angle_indices = {})
    {
        return addTimeStep(time,
            TimeStep<Real, n>(measurement, measurement_jacobian,
            measurement_covariance, measurement_angle_indices));
    }

    /** @brief Gets the state and covariance at a given time.
     *
     *  If there is an existing TimeStep with a close enough time,
     *  it will be used to return the state and covariance.
     *  Otherwise, a new TimeStep with no measurement will be added at the given time,
     *  and the state and covariance will be predicted based on the previous TimeStep.
     *
     *  @param time The time at which to get the state and covariance.
     *  @return A tuple containing the state vector and covariance matrix at time t.
     */
    tuple<State, StateCovariance> getState(Real time) {
        // this call might not add anything and just find an existing TimeStep
        auto step = addNoMeasurement(time);
        return {step.state, step.state_covariance};
    }
    tuple<State, StateCovariance> getLastState() {
        if(timeline.empty()) {
            return {initial_state, initial_state_covariance};
        }
        auto last = std::prev(timeline.end());
        return {last->second.state, last->second.state_covariance};
    }
private:
    /// @brief Returns the total time span covered by the TimeSteps in the TimeLine.
    Real totalTime() const {
        if(timeline.empty()) {
            return Real(0);
        }
        return timeline.rbegin()->first - timeline.begin()->first;
    }
};

} // namespace EKFNamespace
