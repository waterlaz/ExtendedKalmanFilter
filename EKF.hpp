/* Copyright (c) 2026 Evgeniy Vodolazskiy (waterlaz)  */

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

namespace ekf {

/** @brief Computes the inverse CDF of the standard normal distribution.
 *
 *  This function uses Peter J. Acklam's approximation for the inverse CDF
 *  of the normal distribution, which is accurate to about 1.15e-9 for all p in (0,1).
 *
 *  @tparam Real The floating point type to use (e.g. float, double).
 *  @param p The probability value for which to compute the inverse CDF,
 *  must be in the range (0,1).
 *  @return The number z such that the probability of a standard normal random
 *  variable being less than or equal to z is p.
 *  @throws std::domain_error if p is not in the range (0,1).
 */
template<typename Real>
Real normalInverseCDF(Real p) {
    assert(p > 0 && p < 1);

    static const Real a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const Real b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const Real c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const Real d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };

    if (p < 0.02425) {
        Real q = std::sqrt(-2 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
    } else if (p > 1 - 0.02425) {
        Real q = std::sqrt(-2 * std::log(1 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
    } else {
        Real q = p - 0.5;
        Real r = q * q;
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
    }
}

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

/** @brief Checks if a matrix is positive definite.
 *
 *  This function checks if a given square matrix is positive definite
 *  by verifying that it is symmetric and that its LDLT decomposition
 *  is successful and indicates positive definiteness.
 *  An empty matrix (0x0) is considered positive definite by definition.
 *
 *  @tparam Real The floating point type of the matrix elements (e.g. float, double).
 *  @tparam n The dimension of the square matrix (can be set to Dynamic).
 *  @param A The matrix to check for positive definiteness.
 *  @return true if the matrix is positive definite, false otherwise.
 */
template <typename Real, int n>
bool isMatrixPositiveDefinite(const Eigen::Matrix<Real, n, n>& A) {
    if constexpr (n == 0) {
        return true; // an empty matrix is considered positive definite
    } else {
        if(A.rows() != A.cols() || !A.isApprox(A.transpose())) {
            return false;
        }
        Eigen::LDLT<Eigen::Matrix<Real, n, n>> ldlt(A);
        return ldlt.info() == Eigen::Success && ldlt.isPositive();
    }
}

template <typename R, int n, int m>
class GenericMeasurementModel {
public:
    using Real = R;
    using State = Eigen::Matrix<Real, n, 1>;
    using StateCovariance = Eigen::Matrix<Real, n, n>;
    using Measurement = Eigen::Matrix<Real, m, 1>;
    using MeasurementJacobian = Eigen::Matrix<Real, m, n>;
    using MeasurementCovariance = Eigen::Matrix<Real, m, m>;
    /**  @brief The function that maps the state to the tuple (measurement, Jacobian).
     *
     *  This function takes a state vector as input and returns a tuple containing:
     *  - The expected measurement vector corresponding to the input state.
     *  - The measurement Jacobian matrix,
     *  which is the derivative of the measurement function with respect to the state.
     *
     *  This is useful for non-linear measurements.
     */
    static std::pair<Measurement, MeasurementJacobian> measure(const State& state) = delete;
    /**  @brief Indices of the angles in the measurement vector.
     *
     *  Any measurement variable that represents an angle (e.g. orientation)
     *  should have its index included in this vector.
     *  When an angle is updated, it will be wrapped to the range [-pi, pi].
     *  For example, if the state vector is [x, y, theta] and theta is an angle,
     *  then anglesIndices should be set to {2} (assuming 0-based indexing).
     */
    static std::array<size_t, 0> measurementAngleIndices() {
        return {};
    }
    /// @brief The gating probability for outlier rejection based on the
    /// Mahalanobis distance.
    static constexpr Real gatingProbability = 0.99;
};

template <typename Real, int n>
class NoMeasurementModel : public GenericMeasurementModel<Real, n, 0> {
public:
    static std::pair<Eigen::Matrix<Real, 0, 1>, Eigen::Matrix<Real, 0, n>>
        measure(const Eigen::Matrix<Real, n, 1>&)
    {
        return {Eigen::Matrix<Real, 0, 1>(), Eigen::Matrix<Real, 0, n>()};
    }
};

template <typename R, int n>
class GenericProcessModel {
public:
    using Real = R;
    using State = Eigen::Matrix<Real, n, 1>;
    using StateCovariance = Eigen::Matrix<Real, n, n>;
    using StateJacobian = Eigen::Matrix<Real, n, n>;
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
    static std::tuple<State, StateJacobian, StateCovariance> predict(
        const State& state, Real dt) = delete;
};

template <typename Real, int n>
class FilterState {
public:
    using State = Eigen::Matrix<Real, n, 1>;
    using StateCovariance = Eigen::Matrix<Real, n, n>;
    using StateTransition = Eigen::Matrix<Real, n, n>;
    /// @brief Indicates whether the filter state has a valid estimated state and covariance.
    bool hasEstimatedState = false;
    /// @brief The state vector at the time of the observation (x).
    State state;
    /// @brief The covariance matrix of the state estimate (P).
    StateCovariance state_covariance;
    /**
     * @brief Constructs a filter state with an optional state.
     * @param state The state vector (x).
     * @param state_covariance The covariance matrix of the state estimate (P).
     */
    FilterState(const State& state = undefined_state(),
             const StateCovariance& state_covariance = undefined_state_covariance())
        : state{state},
          state_covariance{state_covariance}
    {
        hasEstimatedState = !state.hasNaN();
        assert((!hasEstimatedState || (
                    !state_covariance.hasNaN() &&
                    isMatrixPositiveDefinite(state_covariance))) &&
               "If the state is defined, the covariance must also be defined");
    }
    void predict(const State& predicted_state,
                 const StateCovariance& previous_state_covariance,
                 const StateTransition& transition_jacobian,
                 const StateCovariance& process_noise_covariance)
    {
        const auto& F = transition_jacobian;
        const auto& Q = process_noise_covariance;
        const auto& P_prev = previous_state_covariance;

        assert(isMatrixPositiveDefinite(Q));
        assert(isMatrixPositiveDefinite(P_prev));

        StateCovariance P_pred = F * sym(P_prev) * F.transpose() + Q;
        state = predicted_state;
        state_covariance = P_pred;
    }
private:
    static StateCovariance undefined_state_covariance() {
        return StateCovariance::Constant(std::numeric_limits<Real>::quiet_NaN());
    }
    static State undefined_state() {
        return State::Constant(std::numeric_limits<Real>::quiet_NaN());
    }
    // a usefull optimisation to treat symmetric matrices in computations
    // also improves numerical stability.
    template <typename Derived>
    static inline auto sym(const Eigen::MatrixBase<Derived>& m) {
        return m.template selfadjointView<Eigen::Lower>();
    }
};

/*! A discrete point in time of a Kalman filter.
 *  Most of the filter math is hidden within this class in the update() method.
 *  Each TimeStep can represent either a prediction step (with no measurement)
 *  or an update step (with a measurement).
 *  The EKF class manages a timeline of these TimeSteps,
 *  and performs the necessary prediction and update steps.
 *
 *  The TimeStep class is designed to be flexible and can work with
 *  any measurement model that defines the necessary types and functions.
 * */
template <typename MeasurementModel>
class MeasurementStep {
public:
    using Real = typename MeasurementModel::Real;
    static constexpr int n = MeasurementModel::State::RowsAtCompileTime;
    using State = typename MeasurementModel::State;
    using StateCovariance = typename MeasurementModel::StateCovariance;
    using Measurement = typename MeasurementModel::Measurement;
    using MeasurementJacobian = typename MeasurementModel::MeasurementJacobian;
    using MeasurementCovariance = typename MeasurementModel::MeasurementCovariance;
    using KalmanGain = Eigen::Matrix<Real, State::RowsAtCompileTime,
                                    Measurement::RowsAtCompileTime>;
    /// @brief The time associated with the measurement.
    Real time;
    /// @brief The measurement vector associated with the observation (z).
    Measurement measurement;
    ///  @brief The covariance matrix of the measurement noise (R).
    MeasurementCovariance measurement_covariance;
    /**
     * FIXME: write comment and check if this is correct?
     */
    MeasurementStep() {}
    /**
     * @brief Constructs an observation with a given measurement,
     * measurement function and measurement noise covariance.
     * @param measurement The actual measurement.
     * @param measurement_covariance The covariance matrix of the measurement noise.
     */
    MeasurementStep(const Measurement& measurement,
                    const MeasurementCovariance& measurement_covariance)
        : measurement{measurement},
          measurement_covariance{measurement_covariance}
    {
        assert(isMatrixPositiveDefinite(measurement_covariance));
        assert(measurement.size() == measurement_covariance.rows());
    }

    /**
     * @brief Performs the EKF update step for this measurement.
     * @param filter_state Contains the predicted state and covariance to update.
     */
    void update(FilterState<Real, n>& filter_state) {
        StateCovariance I = StateCovariance::Identity();
        const auto& R = measurement_covariance;
        const auto& x_pred = filter_state.state;
        const auto& P_pred = filter_state.state_covariance;

        assert(isMatrixPositiveDefinite(R));

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
            Eigen::LDLT<MeasurementCovariance> ldlt(S);
            if(ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
                // try to recover and fix the covariance matrix
                filter_state.state_covariance = 0.5*(P_pred + P_pred.transpose());
                Real min_diag = filter_state.state_covariance.diagonal().minCoeff();
                filter_state.state_covariance += 2.0*std::abs(min_diag) * I;
                return;
            }
            Real threshold = getMahalanobisThreshold();
            if(threshold > 0) {
                Real mahalanobisDist = y.transpose() * ldlt.solve(y);
                if(mahalanobisDist > threshold) {
                    // the measurement is too far from the prediction,
                    // we ignore it and just use the predicted state and covariance.
                    return;
                }
            }

            //Mat K = P_pred * H.transpose() * S.inverse(); but stable:
            KalmanGain K = ldlt.solve(H * P_pred).transpose();

            filter_state.state = x_pred + K * y;
            // state_covariance = (I - K * H) * P_pred; but in Joseph form:
            filter_state.state_covariance = (I - K * H) * sym(P_pred) * (I - K * H).transpose()
                             + K * R * K.transpose();
            filter_state.state_covariance = sym(filter_state.state_covariance);
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
        return measurement.size();
    }
private:
    static Measurement no_measurement() {
        return Measurement();
    }
    static MeasurementCovariance no_measurement_covariance() {
        return MeasurementCovariance();
    }
    static MeasurementJacobian no_measurement_jacobian() {
        return MeasurementJacobian();
    }
    // a usefull optimisation to treat symmetric matrices in computations
    // also improves numerical stability.
    template <typename Derived>
    static inline auto sym(const Eigen::MatrixBase<Derived>& m) {
        return m.template selfadjointView<Eigen::Lower>();
    }
    Real getMahalanobisThreshold() {
        // a static variable to cache the computed threshold value.
        static const Real value = computeMahalanobis();
        return value;
    }
    // a threshold for outlier rejection based on the
    // chi-squared distribution with m degrees of freedom and 99% confidence level.
    Real computeMahalanobis() {
        Real m = measurement.size() > 0 ? measurement.size() : 1;
        Real z = normalInverseCDF(MeasurementModel::gatingProbability);
        Real a = 2.0 / (9.0 * m);
        return m * pow(1 - a + z * sqrt(a), 3);
    }
};

template <typename Real, int n, typename... MeasurementModels>
class TimeStepVariant {
public:
    Real time;
    FilterState<Real, n> state;
    std::variant<MeasurementStep<NoMeasurementModel<Real, n>>,
                 MeasurementStep<MeasurementModels>...> measurement_step;
    bool operator<=(const TimeStepVariant<Real, n, MeasurementModels...>& other) const {
        return time<=other.time;
    }
    TimeStepVariant() {}
    TimeStepVariant(Real time) :
        time{time},
        measurement_step{MeasurementStep<NoMeasurementModel<Real, n>>()}
    {}
    TimeStepVariant(Real time, const FilterState<Real, n>& state) :
        time{time},
        state{state},
        measurement_step{MeasurementStep<NoMeasurementModel<Real, n>>()}
    {}
    template<typename MeasurementModel>
    TimeStepVariant(Real time,
                    const MeasurementStep<MeasurementModel>& measurement_step) :
        time{time},
        measurement_step{measurement_step}
    {}
};

template <typename TimeStep>
class TimeLine {
public:
    size_t head = 0;
    size_t tail = 0;
    // returns index of the inserted element
    int insert(const TimeStep& value) {
        if(empty()) {
            data[tail] = value;
            tail = next(tail);
            return prev(tail);
        }
        size_t i = tail;
        while(i!=head) {
            size_t j = prev(i);
            if(data[j] <= value) {
                break;
            }
            data[i] = std::move(data[j]);
            i = j;
        }
        data[i] = value;
        tail = next(tail);
        return i;
    }
    TimeStep& front() {
        return data[head];
    }
    TimeStep& back() {
        return data[prev(tail)];
    }
    void clear() {
        data.clear();
    }
    bool empty() const {
        return head==tail;
    }
    TimeLine(size_t size=0) : data(size) {}
    TimeStep& operator[](size_t i) {
        return data[i];
    }
    size_t prev(size_t index) const {
        return (index + data.size() - 1) % data.size();
    }
    size_t next(size_t index) const {
        return (index + 1) % data.size();
    }
private:
    std::vector<TimeStep> data;
};

template <typename ProcessModel, typename... MeasurementModels>
class EKF {
public:
    static constexpr int n = ProcessModel::State::RowsAtCompileTime;
    using Real = typename ProcessModel::Real;
    using State = typename ProcessModel::State;
    using StateCovariance = typename ProcessModel::StateCovariance;
    using StateJacobian = typename ProcessModel::StateJacobian;
    using TimeStep = TimeStepVariant<Real, n, MeasurementModels...>;
    //using MeasurementModel = typename TimeStep<Real, n>::MeasurementModel;
    /// @brief stores TimeSteps sorted by time.
    TimeLine<TimeStep> timeline;
    /// @brief the initial state vector for the EKF when there are no previous steps.
    State initial_state;
    /// @brief the initial state covariance matrix for the EKF.
    StateCovariance initial_state_covariance;
    /// @brief resets the EKF to the initial state.
    void reset() {
        timeline.clear();
    }
    /** @brief Adds a new TimeStep and performs the EKF prediction and update steps.
     *
     *  The EKF prediction and update steps will be performed for the new
     *  TimeStep and all subsequent TimeSteps in the TimeLine.
     *  @param time The time associated with the new TimeStep to add to the TimeLine.
     *  @param timestep The TimeStep to add to the TimeLine and perform EKF steps for.
     *  @return reference to the added TimeStep in the TimeLine.
     */
    template<typename MeasurementModel>
    const TimeStep& addMeasurementStep(
        Real time,
        const MeasurementStep<MeasurementModel>& measurement_step)
    {
        if(timeline.empty()) {
            size_t i = timeline.insert(TimeStep(
                time,
                FilterState<Real, n>(initial_state, initial_state_covariance)));
        }
        if(timeline.front().time > time) {
            return timeline.front();
        }
        size_t res = timeline.insert(TimeStep(time, measurement_step));
        res = timeline.prev(res);

        for(size_t i = res; i != timeline.tail; i = timeline.next(i)) {
            if(i != timeline.head) {
                auto& prev = timeline[timeline.prev(i)];
                auto& cur = timeline[i];
                Real dt = cur.time - prev.time;
                auto x_prev = prev.state.state;
                auto P_prev = prev.state.state_covariance;
                auto [x_pred, F, Q] = ProcessModel::predict(x_prev, dt);
                cur.state.predict(x_pred, P_prev, F, Q);
                std::visit([&](auto&& step) {
                    step.update(cur.state); }, cur.measurement_step);
            }
        }
        return timeline[res];
    }

    /// @brief Adds a new TimeStep with no measurement at the given time and
    /// performs the EKF prediction step based on the previous TimeStep.
    const TimeStepVariant<Real, n, MeasurementModels...>& addNoMeasurement(Real time) {
        return addMeasurementStep(time, MeasurementStep<NoMeasurementModel<Real, n>>());
    }
    /** @brief Adds a new general TimeStep with a measurement at the given time.
     *
     * @tparam m The dimension of the measurement vector (can be set to Dynamic).
     * @param time The time associated with the measurement.
     * @param measurement The actual measurement vector.
     * @param measurement_covariance The covariance matrix of the measurement noise.
     * @return reference to the added TimeStep in the TimeLine.
     */
    template<typename MeasurementModel>
    const TimeStepVariant<Real, n, MeasurementModels...>& addMeasurement(
        Real time,
        const typename MeasurementModel::Measurement& measurement,
        const typename MeasurementModel::MeasurementCovariance& measurement_covariance)
    {
        return addMeasurementStep(time,
            MeasurementStep<MeasurementModel>(measurement, measurement_covariance));
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
    std::tuple<State, StateCovariance> getState(Real time) {
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
    std::tuple<State, StateCovariance> getLastState() {
        if(timeline.empty()) {
            return {initial_state, initial_state_covariance};
        }
        auto& last = timeline.back();
        return {last.state.state, last.state.state_covariance};
    }
    size_t getMaxHistoryCount() const {
        return timeline.size();
    }
    EKF() : timeline(1000) {}
private:
    /// @brief Returns the total time span covered by the TimeSteps in the TimeLine.
    Real totalTime() const {
        if(timeline.empty()) {
            return Real(0);
        }
        return timeline.rbegin()->time - timeline.begin()->time;
    }
};

} // namespace ekf


