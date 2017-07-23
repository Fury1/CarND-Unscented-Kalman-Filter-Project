#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // If this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // If this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Number of state dimensions
  n_x_ = 5;

  // Augmented state dimensions (add 2 noise components to the state vector n_x_)
  n_aug_ = n_x_ + 2;

  // Initial state vector shape, initialize to all zeros
  x_ = VectorXd::Zero(n_x_);

  // Covariance matrix shape, initialize to an indentity matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Number of sigma points needed to represent the augmented state
  num_sigma_points = 2 * n_aug_ + 1;

  // Predicted Sigma point matrix shape
  Xsig_pred_ = MatrixXd(n_x_, num_sigma_points);

  // Sigma point spreading parameter value
  lambda_ = 3 - n_aug_;

  // Generate the sqrt(lambda_ + n_aug_) once for multiple usage
  sqrt_lambda_n_aug_ = sqrt(lambda_ + n_aug_);

  /*  Weight vector dimensions for predicted mean state and covariance matrix,
  generates the corresponding weights to inverse the predicted sigma point spread */
  weights_ = VectorXd(num_sigma_points);
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  for (int i = 1; i < num_sigma_points; i++) {  // 2n+1 weights
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  /*
  * Check to see if the UKF class has been initialized with starting values,
  * if not set up starting values.
  */
  if (!is_initialized_) {

    // Set previous timestep
    time_us_ = meas_package.timestamp_;

    // Check for a radar or lidar measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      // Convert radar from polar to cartesian coordinates and initialize state
      x_(0) = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      x_(1) = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

      // Initialize state.
      x_(0)= meas_package.raw_measurements_[0];
      x_(1)= meas_package.raw_measurements_[1];
    }

    // Done initializing, switch initialized state to true (no need to predict/update)
    is_initialized_ = true;
  }
  else {

    // Calculate the delta time between previous and current data
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; // delta_t in seconds

    /* Update the current timestamp for next cycle
     and predict the new state using the calculated delta_t*/
    time_us_ = meas_package.timestamp_;
    // Predict the new state
    Prediction(delta_t);

    // Update the state with the correct measurement type
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    }
  }
  // Finish Process Measurement
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /*
  * Generate sigma points (with noise vector augmentation, replaces the Q_ noise matrix)
  * from the current state vector and corresponding covariance matrix (x_, P_)
  */

  // Create an augmented state vector, fill with zeros
  VectorXd x_aug_ = VectorXd::Zero(n_aug_);

  // Create an augmented state covariance, fill with zeros
  MatrixXd P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);

  // Create an augmented sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, num_sigma_points);

  // Insert the current state matrix x_ into the augmented state matrix
  x_aug_.head(n_x_) = x_;

  // Insert the current covariance matrix P_ into the augmented covariance matrix
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;

  /*
  Insert the noise into the lower right 2x2 block of the augmented covariance matrix
  Example of the augmented matrix shape 7x7:
  0,0,0,0,0,0,0
  0,0,0,0,0,0,0
  0,0,0,0,0,0,0
  0,0,0,0,0,0,0
  0,0,0,0,0,0,0
  0,0,0,0,0,1,0 Insertion longitudinal acceleration
  0,0,0,0,0,0,1 Insertion yaw acceleration
  */

  // Insert the process noise standard deviation longitudinal acceleration by indexing in
  P_aug_(n_x_, n_x_) = std_a_ * std_a_; // std_a_^2

  // Insert the process noise standard deviation yaw acceleration by indexing in
  P_aug_(n_aug_ - 1, n_aug_ - 1) = std_yawdd_ * std_yawdd_; // std_yawdd^2

  /*
  * Generate the augmented sigma point matrix
  */

  // Find the square root of matrix P_aug_
  MatrixXd sqrt_p_aug_ = P_aug_.llt().matrixL();

  // Insert the augmented state vector into the first column of the augmented sigma point matrix
  Xsig_aug_.col(0)  = x_aug_;

  // Generate the rest of the augmented sigma point matrix
  for (int i = 0; i < n_aug_; i++) {
    // Columns 1 to n_aug_
    Xsig_aug_.col(i + 1) = x_aug_ + sqrt_lambda_n_aug_ * sqrt_p_aug_.col(i);
    // Columns (n_aug_ + 1) to n_aug_.size()
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt_lambda_n_aug_ * sqrt_p_aug_.col(i);
  }

  /*
  * Predict new sigma points from the augmented sigma point matrix
  *  and add them to the Xsig_pred_ matrix using the CTRV process model
  */
  for (int i = 0; i < num_sigma_points; i++) {

    // Extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    // Predicted state values
    double px_p, py_p;
    //Cos() and Sin() of yaw
    double cos_yaw = cos(yaw);
    double sin_yaw = sin(yaw);

    // Check for division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin (yaw + yawd * delta_t) - sin_yaw);
      py_p = p_y + v / yawd * (cos_yaw - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos_yaw;
      py_p = p_y + v * delta_t * sin_yaw;
    }

    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos_yaw;
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin_yaw;
    v = v + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Put the predicted sigma points into the Xsig_pred_ matrix
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  /*
  * Predict new state vector and corresponding covariance matrix (x_, P_)
  * from the generated sigma points Xsig_pred_
  */

  // Predicted state mean
  x_.fill(0.0); // Clear the state vector
  for (int i = 0; i < num_sigma_points; i++) {  // iterate over sigma points
    x_ += weights_(i) * Xsig_pred_.col(i); // use the weights to invert the sigma points to a mean
  }

  // Predicted state covariance matrix
  P_.fill(0.0); // Clear the covariance matrix
  for (int i = 0; i < num_sigma_points; i++) {  // iterate over sigma points

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
  // Finish Prediction
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

// Dimensions for lidar measurement space (x, y)
int n_z = 2;

//Get the predicted sigma points in a matrix for lidar measurement space
MatrixXd Zsig = Xsig_pred_.topLeftCorner(2, Xsig_pred_.cols());

/* Update the state vector and covariance matrix x_ and P_ with the
  correct measurement formated sigma matrix */
Update(n_z, Zsig, meas_package);
 // Finish lidar update
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // Dimensions for radar measurement space (rho, phi, r_dot)
  int n_z = 3;

  // Create matrix for sigma points in measurement space for radar
  MatrixXd Zsig = MatrixXd(n_z, num_sigma_points);

  // Transform sigma points into measurement space
  for (int i = 0; i < num_sigma_points; i++) {  // 2n+1 simga points

    // Extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model, convert from cartesian coordinates to polar
    Zsig(0,i) = sqrt(p_x * p_x + p_y * p_y); //r
    Zsig(1,i) = atan2(p_y,p_x); //phi
    Zsig(2,i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y); //r_dot
  }

/* Update the state vector and covariance matrix x_ and P_ with the
correct measurement formated sigma matrix */
Update(n_z, Zsig, meas_package);
// Finish radar update
}

/**
* Universal measurement update function for radar and lidar measurements
* Updates the state and covariance matrix (x_, P_) using predicted sigma points
* and a radar or lidar measurement
* @param {int} n_z
* @param {MatrixXd} Zsig
* @param {MeasurementPackage} meas_package
*/
 void  UKF::Update(int n_z, MatrixXd Zsig, MeasurementPackage meas_package) {
  // Mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  // Find the mean predicted measurement
  for (int i = 0; i < num_sigma_points; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z,n_z);

  // Measurement noise matrix R
  MatrixXd R = MatrixXd(n_z,n_z);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    // Generate covariance matrix S
    for (int i = 0; i < num_sigma_points; i++) {  // 2n+1 sigma points
      // Residual
      VectorXd z_diff = Zsig.col(i) - z_pred;
      S += weights_(i) * z_diff * z_diff.transpose();
    }

    // Add measurement noise covariance matrix
    R << std_laspx_* std_laspx_, 0,
         0, std_laspy_ * std_laspy_;
    S += R;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Generate covariance matrix S
    for (int i = 0; i < num_sigma_points; i++) {  // 2n+1 sigma points
      // Residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // Angle normalization
      while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
      while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

      S += weights_(i) * z_diff * z_diff.transpose();
    }

    // Add measurement noise covariance matrix
    R << std_radr_* std_radr_, 0, 0,
         0, std_radphi_* std_radphi_, 0,
         0, 0,std_radrd_* std_radrd_;
    S += R;
  }

   /*
   * Merge the measurement with the prediction and update the state and covariance matrix
   */

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // Calculate cross correlation matrix
  for (int i = 0; i < num_sigma_points; i++) {  // 2n+1 sigma points

    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Get the measurement values from the data as a vector
  VectorXd z = VectorXd(meas_package.raw_measurements_);

  // Residual
  VectorXd z_diff = z - z_pred;

  // Angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // Update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

// Finish measurement update
}