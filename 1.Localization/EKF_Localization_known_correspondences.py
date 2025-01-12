#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Implementation of EKF Localization with known correspondences.
See Probabilistic Robotics:
    1. Page 204, Table 7.2 for full algorithm.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt


class ExtendedKalmanFilter():
    def __init__(self, dataset, end_frame, R, Q):
        self.load_data(dataset, end_frame)
        self.initialization(R, Q)
        for data in self.data:
            subject_num = data[1]
            if (subject_num == -1):
                self.motion_update(data)
            else:
                self.measurement_update(data)
        self.plot_data()

    def load_data(self, dataset, end_frame):
        # Loading dataset
        # Barcodes: [Subject#, Barcode#]
        self.barcodes_data = np.loadtxt(dataset + "/Barcodes.dat")
        # Ground truth: [Time[s], x[m], y[m], orientation[rad]]
        self.groundtruth_data = np.loadtxt(dataset + "/Groundtruth.dat")
        # Landmark ground truth: [Subject#, x[m], y[m]]
        self.landmark_groundtruth_data = np.loadtxt(dataset + "/Landmark_Groundtruth.dat")
        # Measurement: [Time[s], Subject#, range[m], bearing[rad]]
        self.measurement_data = np.loadtxt(dataset + "/Measurement.dat")
        # Odometry: [Time[s], Subject#, forward_V[m/s], angular _v[rad/s]]
        self.odometry_data = np.loadtxt(dataset + "/Odometry.dat")

        # Collect all input data and sort by timestamp
        # Add subject "odom" = -1 for odometry data
        odom_data = np.insert(self.odometry_data, 1, -1, axis = 1) # column 1에 -1을 추가
        self.data = np.concatenate((odom_data, self.measurement_data), axis = 0)
        self.data = self.data[np.argsort(self.data[:, 0])]
        '''
        self.data에는 시간순으로 정렬된 odometry data(-1, v, w)와 measurement data(subject#, r, phi)가 들어있다.
        self.data[:, 0]은 timestamp
        self.data[:, 1]은 subject number
        self.data[:, 2]은 range or forward_V(subject가 -1인 경우)
        self.data[:, 3]은 bearing or angular_v(subject가 -1인 경우)
        '''

        # Remove all data before the first timestamp of groundtruth
        # Use first groundtruth data as the initial location of the robot
        for i in range(len(self.data)):
            if (self.data[i, 0] > self.groundtruth_data[0, 0]):
                break
        self.data = self.data[i:, :]

        # Remove all data after the specified number of frames
        self.data = self.data[:end_frame, :]
        cut_timestamp = self.data[-1, 0]
        # Remove all groundtruth after the corresponding timestamp
        for i in range(len(self.groundtruth_data)):
            if (self.groundtruth_data[i][0] >= cut_timestamp):
                break
        self.groundtruth_data = self.groundtruth_data[:i, :]

        # Combine barcode Subject# with landmark Subject# to create lookup-table
        # [x[m], y[m], x std-dev[m], y std-dev[m]]
        self.landmark_locations = {} # dictionary
        for i in range(5, len(self.barcodes_data), 1):
            subject_num = self.barcodes_data[i, 1]
            self.landmark_locations[subject_num] = self.landmark_groundtruth_data[i - 5, 1:]

        # Lookup table to map barcode Subjec# to landmark Subject#
        # Barcode 6 is the first landmark (1 - 15 for 6 - 20)
        self.landmark_indexes = {}
        for i in range(5, len(self.barcodes_data), 1):
            subject_num = self.barcodes_data[i, 1]
            self.landmark_indexes[subject_num] = i - 4

    def initialization(self, R, Q):
        # Initial state
        self.states = np.array([self.groundtruth_data[0, :]])
        # Choose very small process covariance because we are using the ground truth data for initial location
        self.sigma = np.diagflat([1e-10, 1e-10, 1e-10])
        # States with measurement update
        self.states_measurement = []

        # State covariance matrix
        self.R = R
        # Measurement covariance matrix
        self.Q = Q

    def motion_update(self, control):
        # ------------------ Step 1: Mean update ---------------------#
        # State: [x, y, θ]
        # Control: [v, w]
        # State-transition function (simplified):
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t

        # Get last state
        last_state = self.states[-1]
        t_1 = last_state[0]
        x_t_1 = last_state[1]
        y_t_1 = last_state[2]
        th_t_1 = last_state[3]

        # Get control data
        t = control[0]
        subject_num = control[1]
        v_t = control[2]
        w_t = control[3]

        # Skip motion update if two odometry data are too close
        dt = t - t_1
        if (dt < 0.001):
            return
        # Compute updated [x, y, theta]
        x_t_bar = x_t_1 + v_t * np.cos(th_t_1) * dt
        y_t_bar = y_t_1 + v_t * np.sin(th_t_1) * dt
        th_t_bar = th_t_1 + w_t * dt
        # Limit θ within [-pi, pi]
        if (th_t_bar > np.pi):
            th_t_bar -= 2 * np.pi
        elif (th_t_bar < -np.pi):
            th_t_bar += 2 * np.pi
        # append updated state
        updated_state = np.array([[t, x_t_bar, y_t_bar, th_t_bar]])
        self.states = np.append(self.states, updated_state, axis = 0)

        # ------ Step 2: Linearize state-transition by Jacobian ------#
        # Jacobian: G = d g(u_t, x_t-1) / d x_t-1
        #         1  0  -v * dt * sinθ_t-1
        #   G  =  0  1   v * dt * cosθ_t-1
        #         0  0             1
        self.G = np.array([
            [1, 0, -v_t * dt * np.sin(th_t_1)],
            [0, 1, v_t * dt * np.cos(th_t_1)],
            [0, 0, 1],
        ])

        # ---------------- Step 3: Covariance update ------------------#
        self.sigma = self.G.dot(self.sigma).dot(self.G.T) + self.R

    def measurement_update(self, measurement):
        # Measurement: [Time[s], Subject#, range[m], bearing[rad]]
        t = measurement[0]
        subject_num = measurement[1]
        r_t = measurement[2]
        phi_t = measurement[3]

        # Continue if landmark is not found in self.landmark_locations
        if not subject_num in self.landmark_locations:
            return
        
        landmark_location = self.landmark_locations[subject_num]
        x_l = landmark_location[0]
        y_l = landmark_location[1]

        updated_state = self.states[-1]
        t_1 = updated_state[0]
        x_t_bar = updated_state[1]
        y_t_bar = updated_state[2]
        th_t_bar = updated_state[3]

        # ---------------- Step 1: Measurement update -----------------#
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t
        q = (x_l - x_t_bar) **2 + (y_l - y_t_bar) **2
        sqrt_q = np.sqrt(q)
        range_expected = sqrt_q
        bearing_expected = np.arctan2(y_l - y_t_bar, x_l - x_t_bar) - th_t_bar

        # -------- Step 2: Linearize Measurement by Jacobian ----------#
        # Jacobian: H = d h(x_t) / d x_t
        #        -(x_l - x_t) / sqrt(q)   -(y_l - y_t) / sqrt(q)   0
        #  H  =      (y_l - y_t) / q         -(x_l - x_t) / q     -1
        #                  0                         0             0
        #  q = (x_l - x_t)^2 + (y_l - y_t)^2
        self.H = np.array([
            [-(x_l - x_t_bar) / sqrt_q, -(y_l - y_t_bar) / sqrt_q, 0],
            [(y_l - y_t_bar) / q, -(x_l - x_t_bar) / q, -1],
            [0, 0, 0],
        ])

        # ---------------- Step 3: Kalman gain update -----------------#
        S_t = self.H.dot(self.sigma).dot(self.H.T) + self.Q
        self.K = self.sigma.dot(self.H.T).dot(np.linalg.inv(S_t))

        # ------------------- Step 4: mean update ---------------------#
        difference = np.array([r_t - range_expected, phi_t - bearing_expected, 0])
        innovation = self.K.dot(difference)
        self.states = np.append(self.states, np.array([[t, x_t_bar + innovation[0], y_t_bar + innovation[1], th_t_bar + innovation[2]]]), axis=0)
        self.states_measurement.append([x_t_bar + innovation[0], y_t_bar + innovation[1]])

        # ---------------- Step 5: covariance update ------------------#
        self.sigma = (np.identity(3) - self.K.dot(self.H)).dot(self.sigma)

    def plot_data(self):
        # Ground truth data
        plt.plot(self.groundtruth_data[:, 1], self.groundtruth_data[:, 2], 'b', label="Robot State Ground truth")

        # States
        plt.plot(self.states[:, 1], self.states[:, 2], 'r', label="Robot State Estimate")

        # Start and end points
        plt.plot(self.groundtruth_data[0, 1], self.groundtruth_data[0, 2], 'go', label="Start point")
        plt.plot(self.groundtruth_data[-1, 1], self.groundtruth_data[-1, 2], 'yo', label="End point")

        # Measurement update locations
        if (len(self.states_measurement) > 0):
            self.states_measurement = np.array(self.states_measurement)
            plt.scatter(self.states_measurement[:, 0], self.states_measurement[:, 1], s=10, c='k', alpha=0.5, label="Measurement updates")

        # Landmark ground truth locations and indexes
        landmark_xs = []
        landmark_ys = []
        for location in self.landmark_locations:
            landmark_xs.append(self.landmark_locations[location][0])
            landmark_ys.append(self.landmark_locations[location][1])
            index = self.landmark_indexes[location] + 5
            plt.text(landmark_xs[-1], landmark_ys[-1], str(index), alpha=0.5, fontsize=10)
        plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2, marker='*', label='Landmark Locations')

        # plt.title("Localization with only odometry data")
        plt.title("EKF Localization with Known Correspondences")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # # Dataset 0
    # dataset = "../0.Dataset0"
    # end_frame = 10000
    # # State covariance matrix
    # R = np.diagflat(np.array([1.0, 1.0, 1.0])) ** 2
    # # Measurement covariance matrix
    # Q = np.diagflat(np.array([350, 350, 1e16])) ** 2

    # Dataset 1
    dataset = "0.Dataset1"
    end_frame = 3200
    # State covariance matrix
    R = np.diagflat(np.array([1.0, 1.0, 10.0])) ** 2
    # Measurement covariance matrix
    Q = np.diagflat(np.array([30, 30, 1e16])) ** 2
    #
    ekf = ExtendedKalmanFilter(dataset, end_frame, R, Q)

    a=1
