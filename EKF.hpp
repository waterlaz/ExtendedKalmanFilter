/* Copyright (c) 2026 Evgeniy Vodolazskiy (waterlaz)  */

#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <functional>
#include <map>
#include <tuple>
#include <variant>
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
bool isMatrixPositiveDefinite(const Matrix<Real, n, n>& A) {
    if constexpr (n == 0) {
        return true; // an empty matrix is considered positive definite
    } else {
        if(A.rows() != A.cols() || !A.isApprox(A.transpose())) {
            return false;
        }
        LDLT<Matrix<Real, n, n>> ldlt(A);
        return ldlt.info() == Success && ldlt.isPositive();
    }
}

// a threshold for outlier rejection based on the
// chi-squared distribution with m degrees of freedom and 99% confidence level.
//Real chi2_threshold(int m) {
//    Real z = Real(2.326);
//    Real a = 2.0 / (9.0 * m);
//    return m * pow(1 - a + z * sqrt(a), 3);
//}

template <typename R, int n, int m>
class GenericMeasurementModel {
public:
    using Real = R;
    using State = Matrix<Real, n, 1>;
    using StateCovariance = Matrix<Real, n, n>;
    using StateTransition = Matrix<Real, n, n>;
    using Measurement = Matrix<Real, m, 1>;
    using MeasurementJacobian = Matrix<Real, m, n>;
    using MeasurementCovariance = Matrix<Real, m, m>;
    /**  @brief The function that maps the state to the tuple (measurement, Jacobian).
     *
     *  This function takes a state vector as input and returns a tuple containing:
     *  - The expected measurement vector corresponding to the input state.
     *  - The measurement Jacobian matrix,
     *  which is the derivative of the measurement function with respect to the state.
     *
     *  This is useful for non-linear measurements.
     */
    static std::pair<Measurement, MeasurementJacobian> measure(const State& state) {
        // implement the measurement model here
        return {};
    }
    /**  @brief Indices of the angles in the measurement vector.
     *
     *  Any measurement variable that represents an angle (e.g. orientation)
     *  should have its index included in this vector.
     *  When an angle is updated, it will be wrapped to the range [-pi, pi].
     *  For example, if the state vector is [x, y, theta] and theta is an angle,
     *  then anglesIndices should be set to {2} (assuming 0-based indexing).
     */
    static vector<int> measurementAngleIndices() {
        return {};
    }
    /// @brief A threshold for outlier rejection based on the Mahalanobis distance.
    static Real threashold() {
        return -1.0;
    }
};

template <typename Real, int n>
class NoMeasurementModel : public GenericMeasurementModel<Real, n, 0> {
public:
    static std::pair<Matrix<Real, 0, 1>, Matrix<Real, 0, n>> measure(const Matrix<Real, n, 1>&) {
        return {Matrix<Real, 0, 1>(), Matrix<Real, 0, n>()};
    }
};

/*! A discrete point in time of a Kalman filter.
 *  Most of the filter math is hidden within this class in the predict() method.
 *  @tparam Real The floating point type to use (e.g. float, double)
 *  @tparam n The dimension of the state vector (can be set to Dynamic)
 * */
template <typename MeasurementModel>
class TimeStep {
public:
    using Real = typename MeasurementModel::Real;
    using State = typename MeasurementModel::State;
    using StateCovariance = typename MeasurementModel::StateCovariance;
    using StateTransition = typename MeasurementModel::StateTransition;
    using Measurement = typename MeasurementModel::Measurement;
    using MeasurementJacobian = typename MeasurementModel::MeasurementJacobian;
    using MeasurementCovariance = typename MeasurementModel::MeasurementCovariance;
    using KalmanGain = Matrix<Real, State::RowsAtCompileTime,
                                    Measurement::RowsAtCompileTime>;
    /// @brief The state vector at the time of the observation (x).
    State state;
    /// @brief The covariance matrix of the state estimate (P).
    StateCovariance state_covariance;
    /// @brief The measurement vector associated with the observation (z).
    Measurement measurement;
    ///  @brief The covariance matrix of the measurement noise (R).
    MeasurementCovariance measurement_covariance;
    /**
     * @brief Constructs an observation with an optional state and no measurement.
     * @param time The time associated with the observation.
     */
    TimeStep(const State& state = undefined_state(),
             const StateCovariance& state_covariance = undefined_state_covariance())
        : state{state},
          state_covariance{state_covariance},
          measurement{no_measurement()},
          measurement_covariance{no_measurement_covariance()} {}
    /**
     * @brief Constructs an observation with a given measurement,
     * measurement function and measurement noise covariance.
     * @param measurement The actual measurement.
     * @param measurement_covariance The covariance matrix of the measurement noise.
     */
    TimeStep(const Measurement& measurement,
             const MeasurementCovariance& measurement_covariance)
        : measurement{measurement},
          measurement_covariance{measurement_covariance}
    {
        assert(isMatrixPositiveDefinite(measurement_covariance));
        assert(measurement.size() == measurement_covariance.rows());
    }

    /**
     * @brief The EKF prediction and update steps based on the previous TimeStep.
     * @param previous The previous TimeStep containing the prior state and covariance.
     * @param process_noise_covariance The covariance matrix of the process noise.
     * @param transition_jacobian The Jacobian matrix of the state transition function.
     * @param predicted_state The predicted state vector based on the state transition function.
     */
    void update(const State& predicted_state,
                const StateCovariance& previous_state_covariance,
                const StateTransition& transition_jacobian,
                const StateCovariance& process_noise_covariance)
    {
        StateCovariance I = StateCovariance::Identity();
        const auto& F = transition_jacobian;
        const auto& Q = process_noise_covariance;
        const auto& P_prev = previous_state_covariance;
        const auto& R = measurement_covariance;
        const auto& x_pred = predicted_state;

        assert(isMatrixPositiveDefinite(Q));
        assert(isMatrixPositiveDefinite(R));
        assert(isMatrixPositiveDefinite(P_prev));

        StateCovariance P_pred = F * sym(P_prev) * F.transpose() + Q;
        state = x_pred;
        state_covariance = P_pred;
        if constexpr (Measurement::RowsAtCompileTime != 0) {
        if(hasMeasurement()) {
            auto [z_pred, H] = MeasurementModel::measure(x_pred);
            assert(z_pred.size() == measurement.size());
            assert(H.rows() == measurement.size());
            Measurement y = measurement - z_pred;
            for(int i : MeasurementModel::measurementAngleIndices()) {
                y[i] = normalizeAngle(y[i]);
            }

            MeasurementCovariance S = H * sym(P_pred) * H.transpose() + R;
            LDLT<MeasurementCovariance> ldlt(S);
            if(ldlt.info() != Success || !ldlt.isPositive()) {
                // try to fix the covariance matrix
                state_covariance = 0.5*(P_pred + P_pred.transpose());
                Real min_diag = state_covariance.diagonal().minCoeff();
                state_covariance += std::abs(min_diag) * I;
                return;
            }
            if(MeasurementModel::threashold() > 0) {
                Real mahalanobisDist = y.transpose() * ldlt.solve(y);
                if(mahalanobisDist > MeasurementModel::threashold()) {
                    // the measurement is too far from the prediction,
                    // we ignore it and just use the predicted state and covariance.
                    return;
                }
            }

            //Mat K = P_pred * H.transpose() * S.inverse(); but stable:
            KalmanGain K = ldlt.solve(H * P_pred).transpose();

            state = x_pred + K * y;
            // state_covariance = (I - K * H) * P_pred; but in Joseph form:
            state_covariance = (I - K * H) * sym(P_pred) * (I - K * H).transpose()
                             + K * R * K.transpose();
            state_covariance = sym(state_covariance);
        }
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
        return Measurement();
    }
    static MeasurementCovariance no_measurement_covariance() {
        return MeasurementCovariance();
    }
    static MeasurementJacobian no_measurement_jacobian() {
        return MeasurementJacobian();
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

template <typename Real, int n, typename... MeasurementModels>
class EKF {
public:
    using State = Matrix<Real, n, 1>;
    using StateCovariance = Matrix<Real, n, n>;
    using StateJacobian = Matrix<Real, n, n>;
    using TimeStepVariant = std::variant<TimeStep<NoMeasurementModel<Real, n>>,
                                 TimeStep<MeasurementModels>...>;
    //using MeasurementModel = typename TimeStep<Real, n>::MeasurementModel;
    /// @brief maximum number of TimeSteps to keep in the TimeLine (or 0 to ingore).
    size_t max_history_count = 1000;
    /// @brief maximum period of time to keep in the TimeLine (or negative to ignore).
    Real max_history_time = -1;
    /// @brief stores TimeSteps sorted by time.
    std::multimap<Real, TimeStepVariant> timeline;
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
    template<typename MeasurementModel>
    const TimeStep<MeasurementModel>& addTimeStep(
        Real time,
        const TimeStep<MeasurementModel>& timestep)
    {
        if(timeline.empty()) {
            timeline.insert({time,
                TimeStep<NoMeasurementModel<Real, n>>(initial_state,
                                                      initial_state_covariance)});
        }
        auto res = timeline.insert({time, timestep});
        for(auto cur = res; cur != timeline.end(); cur++) {
            if(cur != timeline.begin()) {
                auto prev = std::prev(cur);
                Real dt = cur->first - prev->first;
                std::visit([&](auto&& step) {
                    auto [x_prev, P_prev] = getStateAndCovariance(prev->second);
                    auto [x_pred, F, Q] = predict(x_prev, dt);
                    step.update(x_pred, P_prev, F, Q); }, cur->second);
            }
        }
        // remove old timesteps if we exceed the limits
        while( timeline.begin() != res && // explicitly forbid removing the new item
               std::next(timeline.begin()) != res && // and the one before it
               ((max_history_count>0 && timeline.size() > max_history_count) ||
                (max_history_time>0 && totalTime() > max_history_time)) ) {
            timeline.erase(timeline.begin());
        }
        return std::get<TimeStep<MeasurementModel>>(res->second);
    }

    /// @brief Adds a new TimeStep with no measurement at the given time and
    /// performs the EKF prediction step based on the previous TimeStep.
    const TimeStep<NoMeasurementModel<Real, n>>& addNoMeasurement(Real time) {
        return addTimeStep(time, TimeStep<NoMeasurementModel<Real, n>>());
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
    template<typename MeasurementModel>
    const TimeStep<MeasurementModel>& addMeasurement(
        Real time,
        const typename MeasurementModel::Measurement& measurement,
        const typename MeasurementModel::MeasurementCovariance& measurement_covariance)
    {
        return addTimeStep(time,
            TimeStep<MeasurementModel>(measurement, measurement_covariance));
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
    /** @brief Gets the last state and covariance in the TimeLine.
     *
     *  If the TimeLine is empty, it returns the initial state and covariance.
     *  Otherwise, it returns the state and covariance of the last TimeStep.
     *
     *  @return A tuple containing the last state vector and covariance matrix.
     */
    tuple<State, StateCovariance> getLastState() {
        if(timeline.empty()) {
            return {initial_state, initial_state_covariance};
        }
        auto last = std::prev(timeline.end());
        auto [state, cov] = getStateAndCovariance(last->second);
        return {state, cov};
    }
private:
    /// @brief Returns the total time span covered by the TimeSteps in the TimeLine.
    Real totalTime() const {
        if(timeline.empty()) {
            return Real(0);
        }
        return timeline.rbegin()->first - timeline.begin()->first;
    }
    std::tuple<const State&, const StateCovariance&> getStateAndCovariance(
        const TimeStepVariant& step) {
        return std::visit([](auto&& step) -> std::tuple<const State&, const StateCovariance&> {
            return {step.state, step.state_covariance}; }, step);
    }
};

} // namespace EKFNamespace
