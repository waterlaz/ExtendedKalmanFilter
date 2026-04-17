/* Copyright (c) 2026 Evgeniy Vodolazskiy (waterlaz)  */

#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <functional>
#include <set>
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
    LLT<Matrix<Real, n, n>> llt(A);
    return llt.info() == Success;
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
    /// @brief The time associated with the observation.
    Real time;
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
          measurement_angle_indices{angle_indices}
    {
        assert(isMatrixPositiveDefinite(measurement_covariance));
        assert(measurement.size() == measurement_covariance.rows());
    }
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
             const vector<int>& measurement_angle_indices = {})
        : time{time},
          measurement{measurement},
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
                const StateCovariance& process_noise_covariance,
                const StateTransition& transition_jacobian,
                const State& predicted_state)
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

/** @brief Comparison operator for TimeStep based on time. */
template<typename Real, int n>
bool operator<(const TimeStep<Real, n>& a, const TimeStep<Real, n>& b) {
    return a.time < b.time;
}

/// @brief Concatenates two matrices vertically (stacks them on top of each other).
template <typename Real, int K>
Matrix<Real, Dynamic, K> concatVer(const Matrix<Real, Dynamic, K>& A,
                                   const Matrix<Real, Dynamic, K>& B)
{
    assert(A.cols() == B.cols());
    Matrix<Real, Dynamic, K> result(A.rows() + B.rows(), A.cols());
    result << A,
              B;
    return result;
}

/// @brief Concatenates two matrices diagonally (places them in the top-left and bottom-right corners, filling the rest with zeros).
template <typename Real>
Matrix<Real, Dynamic, Dynamic> concatDiag(const Matrix<Real, Dynamic, Dynamic>& A,
                                          const Matrix<Real, Dynamic, Dynamic>& B)
{
    assert(A.cols() == A.rows() && B.cols() == B.rows());
    Matrix<Real, Dynamic, Dynamic> result(A.rows() + B.rows(), A.cols() + B.cols());
    result << A, Matrix<Real, Dynamic, Dynamic>::Zero(A.rows(), B.cols()),
              Matrix<Real, Dynamic, Dynamic>::Zero(B.rows(), A.cols()), B;
    return result;
}

/** @brief Joins two TimeSteps that are close in time into a single TimeStep.
 *
 *  This is useful when multiple measurements are taken at the same time or very close in time,
 *  and we want to treat them as a single measurement for the EKF update step.
 *  The resulting TimeStep will have the time of the first TimeStep.
 *  a concatenated measurement vector, a combined measurement model that
 *  concatenates the outputs of the two input measurement models,
 *  and a block diagonal measurement covariance matrix.
 *
 *  @param a The first TimeStep to join.
 *  @param b The second TimeStep to join.
 *  @return A new TimeStep that combines the measurements of a and b.
 */
template <typename Real, int n>
TimeStep<Real, n> joinTimeSteps(const TimeStep<Real, n>& a,
                                const TimeStep<Real, n>& b)
{
    if(!a.hasMeasurement()) {
        // a is virtually empty, so we can just take b
        return b;
    }
    if(!b.hasMeasurement()) {
        // b is virtually empty, so we can just take a
        return a;
    }
    vector<int> measurement_angle_indices = a.measurement_angle_indices;
    for(int i : b.measurement_angle_indices) {
        measurement_angle_indices.push_back(i + a.measurement.size());
    }
    TimeStep<Real, n> step(
        a.time, // we can take either a.time or b.time since they are close
        concatVer(a.measurement, b.measurement),
        [a_model = a.measurement_model,
         b_model = b.measurement_model](const typename TimeStep<Real, n>::State& state) {
            auto [z_a, H_a] = a_model(state);
            auto [z_b, H_b] = b_model(state);
            Matrix<Real, Dynamic, 1> z = concatVer(z_a, z_b);
            Matrix<Real, Dynamic, Dynamic> H = concatVer(H_a, H_b);
            return std::make_tuple(z, H);
        },
        concatDiag(a.measurement_covariance, b.measurement_covariance),
        measurement_angle_indices);
    if(a.hasEstimatedState()) {
        step.state = a.state;
        step.state_covariance = a.state_covariance;
    } else {
        step.state = b.state;
        step.state_covariance = b.state_covariance;
    }
    return step;
}

/*! A timeline of TimeSteps for the EKF.
 *  @tparam Real The floating point type to use (e.g. float, double)
 *  @tparam n The dimension of the state vector (can be set to Dynamic)
 * */
template <typename Real, int n>
class TimeLine {
public:
    using iterator = typename std::multiset<TimeStep<Real, n>>::iterator;
    /// @brief The maximum time difference between two TimeSteps for them to be considered "close" and thus joined together.
    Real epsilonTime;
    /**  @brief Constructs an empty TimeLine with a given epsilonTime.
     *
     *  epsilonTime is the maximum time difference between two TimeSteps
     *  for them to be considered "close" and thus joined together.
     *  The default value of 1e-9 is suitable for most applications,
     *  but it can be adjusted based on the expected time resolution.
     */
    TimeLine(Real epsilonTime = Real(1e-9)) : epsilonTime{epsilonTime} {}
    /** @brief Inserts a TimeStep into the TimeLine.
     *
     *  If there are existing TimeSteps that are close in time to the new TimeStep,
     *  they will be joined together using the @ref joinTimeSteps function,
     *  and the resulting TimeStep will replace the existing ones.
     *  Measurements taken at the same time or very close in time
     *  are treated as a single measurement for the EKF update step.
     *
     *  @param timestep The TimeStep to insert into the TimeLine.
     *  @return An iterator pointing to the inserted (or joined) TimeStep.
     */
    iterator insert(const TimeStep<Real, n>& timestep) {
        // find timesteps just after the new timestep
        auto after = steps.lower_bound(timestep);
        if(after != steps.end()) {
            if(areClose(*after, timestep)) {
                auto res = steps.emplace_hint(after, joinTimeSteps(*after, timestep));
                steps.erase(after);
                return res;
            }
        }
        if(after != steps.begin()) {
            auto before = std::prev(after);
            if(areClose(*before, timestep)) {
                auto res = steps.emplace_hint(before, joinTimeSteps(*before, timestep));
                steps.erase(before);
                return res;
            }
        }
        return steps.emplace_hint(after, timestep);
    }
    /// @brief Returns an iterator to the beginning of the TimeLine.
    iterator begin() {
        return steps.begin();
    }
    /// @brief Returns an iterator to the end of the TimeLine.
    iterator end() {
        return steps.end();
    }
    /// @brief Clears all TimeSteps from the TimeLine.
    void clear() {
        steps.clear();
    }
    /// @brief Checks if the TimeLine is empty (i.e. contains no TimeSteps).
    bool empty() const {
        return steps.empty();
    }
    /// @brief Returns the number of TimeSteps currently stored in the TimeLine.
    size_t size() const {
        return steps.size();
    }
    /// @brief Returns the total time span covered by the TimeSteps in the TimeLine.
    Real totalTime() const {
        if(steps.empty()) {
            return Real(0);
        }
        return steps.rbegin()->time - steps.begin()->time;
    }
    /// @brief Removes the earliest TimeStep from the TimeLine.
    void pop() {
        if(!steps.empty()) {
            steps.erase(steps.begin());
        }
    }
private:
    /// @brief stores the TimeSteps sorted by time.
    std::multiset<TimeStep<Real, n>> steps;
    /// @brief a helper function to check if two TimeSteps are close in time.
    bool areClose(const TimeStep<Real, n>& a, const TimeStep<Real, n>& b) const {
        return std::abs(a.time - b.time) < epsilonTime;
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
    /// @brief the time precision for joining TimeSteps in the TimeLine.
    /// TimeSteps with time difference less than epsilonTime will be joined together.
    void setTimePrecision(Real epsilonTime) {
        timeline.epsilonTime = epsilonTime;
    }
    /// @brief stores TimeSteps sorted by time.
    TimeLine<Real, n> timeline;
    State initial_state;
    StateCovariance initial_state_covariance;
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
        const State& x, Real dt) = 0;
    /** @brief Adds a new TimeStep and performs the EKF prediction and update steps.
     *
     *  If there already is a TimeStep with a close enough time,
     *  the new timestep will be joined with it using the @ref joinTimeSteps function
     *  and the resulting TimeStep will replace the existing one.
     *  The EKF prediction and update steps will be performed for the new
     *  TimeStep and all subsequent TimeSteps in the TimeLine.
     *  @param timestep The TimeStep to add to the TimeLine and perform EKF steps for.
     *  @return reference to the added (or joined) TimeStep in the TimeLine.
     */
    TimeStep<Real, n>& addTimeStep(const TimeStep<Real, n>& timestep) {
        if(!timeline.empty() && timeline.begin()->time > timestep.time) {
            // do not add old timesteps
            return timeline.begin();
        }
        auto res = timeline.insert(timestep);
        for(auto it = res; it != timeline.end(); it++) {
            if(it == timeline.begin()) {
                continue;
            }
            auto prev = std::prev(it);
            Real dt = it->time - prev->time;
            auto [x_pred, F, Q] = predict(prev->state, dt);
            it->update(*prev, Q, F, x_pred);
        }
        // remove old timesteps if we exceed the limits
        while( timeline.begin() != res && // explicitly forbid removing the new item
               std::next(timeline.begin()) != res && // and the one before it
               ((max_history_count>0 && timeline.size() > max_history_count) ||
                (max_history_time>0 && timeline.totalTime() > max_history_time)) ) {
            timeline.pop();
        }
        return *res;
    }

    /// @brief Adds a new TimeStep with no measurement at the given time and
    /// performs the EKF prediction step based on the previous TimeStep.
    TimeStep<Real, n>& addNoMeasurement(Real time) {
        return addTimeStep(TimeStep<Real, n>(time));
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
    TimeStep<Real, n>& addMeasurement(
                             Real time,
                             const Matrix<Real, m, 1>& measurement,
                             const MeasurementModel& measurement_model,
                             const Matrix<Real, m, m>& measurement_covariance,
                             const vector<int>& measurement_angle_indices = {})
    {
        return addTimeStep(TimeStep<Real, n>(
            time, measurement, measurement_model, measurement_covariance,
            measurement_angle_indices));
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
    TimeStep<Real, n>& addMeasurement(
                             Real time,
                             const Matrix<Real, m, 1>& measurement,
                             const Matrix<Real, m, n>& measurement_jacobian,
                             const Matrix<Real, m, m>& measurement_covariance,
                             const vector<int>& measurement_angle_indices = {})
    {
        return addTimeStep(TimeStep<Real, n>(
            time, measurement, measurement_jacobian, measurement_covariance,
            measurement_angle_indices));
    }

    /** @brief Gets the state and covariance at a given time.
     *
     *  If there is an existing TimeStep with a close enough time,
     *  it will be used to return the state and covariance.
     *  Otherwise, a new TimeStep with no measurement will be added at the given time,
     *  and the state and covariance will be predicted based on the previous TimeStep.
     *
     *  @param t The time at which to get the state and covariance.
     *  @return A tuple containing the state vector and covariance matrix at time t.
     */
    tuple<State, StateCovariance> getState(Real t) {
        // this call might not add anything and just find an existing TimeStep
        auto step = addNoMeasurement(t);
        return {step.state, step.state_covariance};
    }
};

} // namespace EKFNamespace
