def propagationStep1(pose_init, time_init, time_new, left_vel, right_vel, radius, width, left_var, right_var):
    particle_set = []
    change_time = time_new - time_init

    for index in pose_init:
        index = np.array(index)
        left_vel_error = left_vel + np.random.normal(0, left_var)
        right_vel_error = right_vel + np.random.normal(0, right_var)

        omega_dot = [[0, -radius/width*(right_vel_error-left_vel_error), radius/2*(
            right_vel_error+left_vel_error)], [radius/width*(right_vel_error-left_vel_error), 0, 0], [0, 0, 0]]
        omega_dot = np.array(omega_dot)
        particle_set.append(
            np.dot(index, linalg.expm(change_time * omega_dot)))

    return particle_set
