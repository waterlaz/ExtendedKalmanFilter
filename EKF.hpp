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

#ifdef __cpp_concepts
/**
 * @brief Checks if a type satisfies the requirements of a measurement model for the EKF.
 * @details This concept requires that the type M has the following:
 * - A nested type `State` representing the state vector.
 * - A nested type `Measurement` representing the measurement vector.
 * - A nested type `MeasurementCovariance` representing the covariance matrix of the measurement noise.
 * - A nested type `MeasurementJacobian` representing the Jacobian matrix of the measurement function.
 * - A static member function `measure(const State&)` that returns a pair of the expected measurement and its Jacobian.
 * - A static member function `measurementAngleIndices()` that returns a range of indices corresponding to angle measurements.
 *   Angles will be normalized to the range [-pi, pi] during the update step.
 */
template<typename M>
concept MeasurementModelConcept = requires(
    typename M::State x,
    typename M::Measurement z,
    typename M::MeasurementCovariance R,
    typename M::MeasurementJacobian H
) {
    { M::measure(x) } -> std::same_as<std::pair<typename M::Measurement, typename M::MeasurementJacobian>>;
    { M::measurementAngleIndices() } -> std::ranges::range;
};

/**
 * @brief Checks if a type satisfies the requirements of a process model for the EKF.
 * @details This concept requires that the type P has the following:
 * - A nested type `Duration` representing the time duration type used for prediction steps.
 * - A nested type `State` representing the state vector.
 * - A nested type `StateCovariance` representing the covariance matrix of the state estimate.
 * - A nested type `StateJacobian` representing the Jacobian matrix of the state transition function.
 * - A static member function `predict(const State&, Duration dt)` that takes previous state and passed time duration
 *   and returns a tuple of the predicted state, state Jacobian, and process noise covariance.
 */
template<typename P>
concept ProcessModelConcept = requires(
    typename P::Duration dt,
    typename P::State x,
    typename P::StateCovariance Q,
    typename P::StateJacobian F,
    P process
) {
    { process.predict(x, dt) } -> std::same_as<std::tuple<typename P::State, typename P::StateJacobian, typename P::StateCovariance>>;
};

template<typename T>
concept TimeConcept = requires(T a, T b) {
    { a <= b } -> std::convertible_to<bool>;
    { b - a };
};
#else
//If your compiler produces an error here, comment out the #warning directive
#warning "Concepts are not supported by the compiler. Using fallback."
#define MeasurementModelConcept typename
#define ProcessModelConcept typename
#define TimeConcept typename
#endif

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
 */
template<typename Real>
[[nodiscard]] Real normalInverseCDF(Real p) {
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
 *  It is useful for ensuring that angle measurements and state variables
 *  representing angles remain within a consistent range,
 *  which can help prevent issues with angle discontinuities in the EKF update step.
 *
 *  @tparam Real The floating point type of the angle (e.g. float, double).
 *  @param angle The angle to normalize, in radians.
 *  @return The normalized angle in the range [-pi, pi].
 */
template <typename Real>
[[nodiscard]] Real normalizeAngle(Real angle) {
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
[[nodiscard]] bool isMatrixPositiveDefinite(const Eigen::Matrix<Real, n, n>& A) {
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
/**
 * @brief Base class for measurement models used by the Extended Kalman Filter.
 *
 * A measurement model defines how the system state is mapped to an expected
 * measurement and its Jacobian. Derived classes are expected to provide a
 * static `measure()` function and may optionally override
 * `measurementAngleIndices()` and `gatingProbability`.
 *
 * @tparam R Floating-point type (for example `float` or `double`).
 * @tparam n Dimension of the state vector.
 * @tparam m Dimension of the measurement vector.
 */
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
    [[nodiscard]] static std::array<size_t, 0> measurementAngleIndices() {
        return {};
    }
    /** @brief Probability used to compute the Mahalanobis distance gating threshold.
     *
     * Measurements whose innovation Mahalanobis distance exceeds the threshold
     * corresponding to this probability are treated as outliers and ignored.
     */
    static constexpr Real gatingProbability = 0.99;
};

/**
 * @brief Dummy measurement model representing the absence of measurement.
 *
 * This model has zero measurement dimension and is used for prediction-only steps
 *
 * @tparam Real Floating-point type.
 * @tparam n Dimension of the state vector.
 */
template <typename Real, int n>
class NoMeasurementModel : public GenericMeasurementModel<Real, n, 0> {
public:
    [[nodiscard]] static std::pair<Eigen::Matrix<Real, 0, 1>, Eigen::Matrix<Real, 0, n>>
        measure(const Eigen::Matrix<Real, n, 1>&)
    {
        return {Eigen::Matrix<Real, 0, 1>(), Eigen::Matrix<Real, 0, n>()};
    }
};

/**
 * @brief Base class for process models used by the Extended Kalman Filter.
 *
 * A process model predicts the next system state given the current state and
 * elapsed time, and provides the associated state-transition Jacobian and
 * process noise covariance.
 *
 * @tparam R Floating-point type (for example `float` or `double`).
 * @tparam n Dimension of the state vector.
 */
template <typename R, int n, TimeConcept T = R>
class GenericProcessModel {
public:
    using Real = R;
    using Time = T;
    using Duration = decltype(Time() - Time());
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
    [[nodiscard]] std::tuple<State, StateJacobian, StateCovariance> predict(
        const State& state, Duration dt) = delete;
};


/**
 * @brief Stores the current state estimate and its covariance.
 *
 * This class encapsulates the estimated state vector and corresponding
 * covariance matrix and provides functionality for the prediction step.
 *
 * @tparam Real Floating-point type.
 * @tparam n Dimension of the state vector.
 */
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
    /// @brief Constructs a filter state with no state estimate.
    FilterState() = default;
    /**
     * @brief Constructs a filter state with a known estimated state and covariance.
     * @param state The state vector (x).
     * @param state_covariance The covariance matrix of the state estimate (P).
     */
    FilterState(const State& state, const StateCovariance& state_covariance) :
        hasEstimatedState{true},
        state{state},
        state_covariance{state_covariance}
    {
        assert(isMatrixPositiveDefinite(state_covariance) &&
               "If the state is defined, the covariance must also be defined");
    }
    /**
     * @brief Performs the covariance prediction step.
     *
     * @param predicted_state Predicted state vector.
     * @param previous_state_covariance Covariance of the previous state estimate.
     * @param transition_jacobian State-transition Jacobian.
     * @param process_noise_covariance Process noise covariance matrix.
     */
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
        state_covariance = sym(P_pred);
    }
private:
    // a usefull optimisation to treat symmetric matrices in computations
    // also improves numerical stability.
    template <typename Derived>
    static inline auto sym(const Eigen::MatrixBase<Derived>& m) {
        return m.template selfadjointView<Eigen::Lower>();
    }
};

/**
 * @brief Represents a single measurement update in the filter timeline.
 *
 * Stores a measurement vector and its covariance and performs the Extended
 * Kalman Filter update step.
 *
 * @tparam MeasurementModel Measurement model type used to interpret the
 *         measurement.
 */
template <MeasurementModelConcept MeasurementModel>
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
    /// @brief The measurement vector associated with the observation (z).
    Measurement measurement;
    ///  @brief The covariance matrix of the measurement noise (R).
    MeasurementCovariance measurement_covariance;
    /**
     * @brief Constructs an empty measurement step.
     *
     * The measurement vector and covariance matrix are default-constructed.
     * For zero-dimensional measurements (used by NoMeasurementModel), this
     * represents a prediction-only step.
     */
    MeasurementStep() = default;

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
                Real fix = 1e-9 * P_pred.trace() + std::abs(min_diag);
                filter_state.state_covariance += fix * I;
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
    [[nodiscard]] bool hasMeasurement() const {
        return measurement.size() > 0;
    }
private:
    // a usefull optimisation to treat symmetric matrices in computations
    // also improves numerical stability.
    template <typename Derived>
    static inline auto sym(const Eigen::MatrixBase<Derived>& m) {
        return m.template selfadjointView<Eigen::Lower>();
    }
    /**
     * @brief Returns the cached Mahalanobis-distance gating threshold.
     *
     * The threshold is computed once per template instantiation using
     * `computeMahalanobis()`.
     *
     * @return Squared Mahalanobis-distance threshold.
     */
    [[nodiscard]] Real getMahalanobisThreshold() {
        // a static variable to cache the computed threshold value.
        static const Real value = computeMahalanobis();
        return value;
    }
    // a threshold for outlier rejection based on the
    // chi-squared distribution with m degrees of freedom and 99% confidence level.
    [[nodiscard]] Real computeMahalanobis() {
        Real m = measurement.size() > 0 ? measurement.size() : 1;
        Real z = normalInverseCDF(MeasurementModel::gatingProbability);
        Real a = 2.0 / (9.0 * m);
        return m * pow(1 - a + z * sqrt(a), 3);
    }
};

/**
 * @brief Timeline entry containing time, state, and an optional measurement.
 *
 * Each object stores:
 * - The timestamp.
 * - The filter state estimate at that timestamp.
 * - A variant holding either a prediction-only step or one of the supported
 *   measurement step types.
 *
 * @tparam Real Floating-point type.
 * @tparam n Dimension of the state vector.
 * @tparam MeasurementModels Supported measurement model types.
 */
template <typename Real, int n, TimeConcept Time, MeasurementModelConcept... MeasurementModels>
class TimeStepVariant {
public:
    Time time;
    FilterState<Real, n> state;
    std::variant<MeasurementStep<NoMeasurementModel<Real, n>>,
                 MeasurementStep<MeasurementModels>...> measurement_step;
    /**
     * @brief Compares two timeline entries by timestamp.
     *
     * @param other Entry to compare against.
     * @return `true` if this entry occurs no later than `other`.
     */
    [[nodiscard]] bool operator<=(
        const TimeStepVariant<Real, n, Time, MeasurementModels...>& other) const
    {
        return time<=other.time;
    }
    /** @brief Updates the filter state using the stored measurement step.
     *
     * This function applies the EKF update step for the measurement stored in
     * this timeline entry, modifying the provided filter state accordingly.
     * If the measurement step contains no measurement,
     * the filter state will remain unchanged.
     * @param filter_state The filter state to update based on the measurement step.
     */
    void update(FilterState<Real, n>& filter_state) {
        std::visit([&](auto&& step) {
            step.update(filter_state); }, measurement_step);
    }
    /** @brief Constructs an empty timeline entry. */
    TimeStepVariant() = default;
    /**
     * @brief Constructs a prediction-only timeline entry.
     * @param time Timestamp associated with the entry.
     */
    TimeStepVariant(Time time) :
        time{time},
        measurement_step{MeasurementStep<NoMeasurementModel<Real, n>>()}
    {}
    /**
     * @brief Constructs a timeline entry with an explicit filter state.
     * @param time Timestamp associated with the entry.
     * @param state Initial filter state.
     */
    TimeStepVariant(Time time, const FilterState<Real, n>& state) :
        time{time},
        state{state},
        measurement_step{MeasurementStep<NoMeasurementModel<Real, n>>()}
    {}
    /**
     * @brief Constructs a timeline entry containing a measurement step.
     * @tparam MeasurementModel Measurement model type.
     * @param time Timestamp associated with the measurement.
     * @param measurement_step Measurement step object.
     */
    template<MeasurementModelConcept MeasurementModel>
    TimeStepVariant(Time time,
                    const MeasurementStep<MeasurementModel>& measurement_step) :
        time{time},
        measurement_step{measurement_step}
    {}
};

/**
 * @brief Fixed size circular buffer timeline that stores time-ordered filter steps.
 *
 * New entries are inserted in sorted order by timestamp.
 * Existing entries after the insertion point are shifted as needed.
 *
 * @tparam TimeStep Type of entries stored in the timeline.
 */
template <typename TimeStep>
class Timeline {
public:
    /// @brief Handle type used to reference entries in the timeline.
    class Handle {
        friend class Timeline;
        size_t index;
        Handle(size_t index) : index(index) {}
        operator size_t() const { return index; }
    public:
        bool operator==(const Handle& other) const { return index == other.index; }
        bool operator!=(const Handle& other) const { return index != other.index; }
    };
    /// @brief Head handle of the circular buffer.
    Handle head = 0;
    /// @brief Tail handle of the circular buffer.
    Handle tail = 0;
    /**
     * @brief Finds the handle of the last entry in the timeline that is less than or equal to the given value.
     *
     * This function searches the timeline in reverse order, starting from the tail,
     * and returns the handle of the last entry that is less than or equal to the specified value.
     * If no such entry is found, it returns the tail handle, indicating that the value is not present in the timeline.
     * @param value The value to search for in the timeline.
     * @return The handle of the last entry less than or equal to the given value, or the tail handle if not found.
     */
    Handle find(const TimeStep& value) const {
        Handle i = tail;
        while(i != head) {
            Handle j = prev(i);
            if(data[j] <= value) {
                return j;
            }
            i = j;
        }
        return tail; // not found, return tail handle
    }
    /**
     * @brief Inserts a time step while preserving timeline order.
     * @param value Timeline value to insert.
     * @return Index where the value was inserted.
     */
    Handle insert(const TimeStep& value) {
        assert(capacity() > 0 && "Timeline capacity must be greater than 0");
        if(empty()) {
            data[tail] = value;
            tail = next(tail);
            return prev(tail);
        }
        Handle i = tail;
        while(i!=head) {
            Handle j = prev(i);
            if(data[j] <= value) {
                break;
            }
            data[i] = std::move(data[j]);
            i = j;
        }
        data[i] = value;
        tail = next(tail);
        if(tail == head) {
            // the buffer is full, we need to overwrite the oldest entry
            head = next(head);
        }
        return i;
    }
    /// @brief Returns the first element in the timeline.
    [[nodiscard]] TimeStep& front() { return data[head]; }
    [[nodiscard]] const TimeStep& front() const { return data[head]; }
    /// @brief Returns the last element in the timeline.
    [[nodiscard]] TimeStep& back() { return data[prev(tail)]; }
    [[nodiscard]] const TimeStep& back() const { return data[prev(tail)]; }
    /** @brief Clears the timeline by resetting head and tail indices.
     *  The actual data in the vector is not modified, but will be overwritten
     *  by future insertions.
     */
    void clear() {
        head = 0;
        tail = 0;
    }
    /// @brief Checks whether the timeline has no entries.
    [[nodiscard]] bool empty() const {
        return head == tail;
    }
    /// @brief Returns the number of entries currently stored in the timeline.
    [[nodiscard]] size_t size() const {
        return (tail + data.size() - head) % data.size();
    }
    /// @brief Returns the maximum number of entries that can be stored.
    [[nodiscard]] size_t capacity() const {
        return data.size() - 1;
    }
    /// @brief Constructs a timeline with fixed circular capacity.
    Timeline(size_t size=0) : data(size+1) {}
    /// @brief Returns a timeline element by internal handle.
    [[nodiscard]] TimeStep& operator[](Handle i) {
        return data[i];
    }
    /// @brief Returns previous circular handle.
    [[nodiscard]] Handle prev(Handle i) const {
        return (i + data.size() - 1) % data.size();
    }
    /// @brief Returns next circular handle.
    [[nodiscard]] Handle next(Handle i) const {
        return (i + 1) % data.size();
    }
private:
    std::vector<TimeStep> data;
};


/**
 * @brief Extended Kalman Filter with support for multiple measurement models.
 *
 * The filter maintains a time-ordered history of prediction and measurement
 * steps. When a new measurement is inserted at an arbitrary time, all
 * subsequent states are recomputed.
 *
 * @tparam ProcessModel Dynamic model used for state prediction.
 * @tparam MeasurementModels Supported measurement model types.
 */
template <ProcessModelConcept ProcessModel,
          MeasurementModelConcept... MeasurementModels>
class EKF {
public:
    static constexpr int n = ProcessModel::State::RowsAtCompileTime;
    using Real = typename ProcessModel::Real;
    using Time = typename ProcessModel::Time;
    using Duration = typename ProcessModel::Duration;
    using State = typename ProcessModel::State;
    using StateCovariance = typename ProcessModel::StateCovariance;
    using StateJacobian = typename ProcessModel::StateJacobian;
    using TimeStep = TimeStepVariant<Real, n, Time, MeasurementModels...>;
    /// @brief stores TimeSteps sorted by time.
    Timeline<TimeStep> timeline;
    /// @brief the initial state vector for the EKF when there are no previous steps.
    State initial_state;
    /// @brief the initial state covariance matrix for the EKF.
    StateCovariance initial_state_covariance;
    /// @brief the process model used for state prediction.
    ProcessModel process_model;
    /// @brief resets the EKF to the initial state.
    void reset() {
        timeline.clear();
    }
    /** @brief Inserts a new measurement step and updates all subsequent steps.
     *
     *  @param time The time associated with the new TimeStep to add to the TimeLine.
     *  @param timestep The TimeStep to add to the TimeLine and perform EKF steps for.
     *  @return reference to the added TimeStep in the TimeLine.
     */
    template<MeasurementModelConcept MeasurementModel>
    const TimeStep& addMeasurementStep(
        Time time,
        const MeasurementStep<MeasurementModel>& measurement_step)
    {
        if(timeline.empty()) {
            timeline.insert(TimeStep(
                time,
                FilterState<Real, n>(initial_state, initial_state_covariance)));
        }
        if(timeline.front().time > time) {
            return timeline.front();
        }
        if(!measurement_step.hasMeasurement()) {
            auto f = timeline.find(TimeStep(time));
            if(f != timeline.tail && timeline[f].time == time) {
                // if there is already a TimeStep at this time, we just return it
                return timeline[f];
            }
        }
        auto res = timeline.insert(TimeStep(time, measurement_step));

        for(auto i = res; i != timeline.tail; i = timeline.next(i)) {
            if(i != timeline.head) {
                const auto& prev = timeline[timeline.prev(i)];
                auto& cur = timeline[i];
                Duration dt = cur.time - prev.time;
                const auto& x_prev = prev.state.state;
                const auto& P_prev = prev.state.state_covariance;
                auto [x_pred, F, Q] = process_model.predict(x_prev, dt);
                cur.state.predict(x_pred, P_prev, F, Q);
                cur.update(cur.state);
            }
        }
        return timeline[res];
    }

    /// @brief Adds a new TimeStep with no measurement at the given time and
    /// performs the EKF prediction step based on the previous TimeStep.
    const TimeStep& addNoMeasurement(Time time) {
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
    template<MeasurementModelConcept MeasurementModel>
    const TimeStep& addMeasurement(
        Time time,
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
    std::tuple<State, StateCovariance> predictState(Time time) {
        auto step = addNoMeasurement(time);
        return {step.state.state, step.state.state_covariance};
    }
    /** @brief Gets the last state and covariance in the TimeLine.
     *
     *  If the TimeLine is empty, it returns the initial state and covariance.
     *  Otherwise, it returns the state and covariance of the last TimeStep.
     *
     *  @return A tuple containing the last state vector and covariance matrix.
     */
    [[nodiscard]] std::tuple<State, StateCovariance> getLastState() const {
        if(timeline.empty()) {
            return {initial_state, initial_state_covariance};
        }
        auto& last = timeline.back();
        return {last.state.state, last.state.state_covariance};
    }
    /// @brief Returns configured timeline capacity.
    [[nodiscard]] size_t getMaxHistoryCount() const {
        return timeline.capacity();
    }
    /// @brief Constructs EKF with default timeline history capacity.
    explicit EKF(size_t history_capacity=1000) : timeline(history_capacity) {}
};

} // namespace ekf


