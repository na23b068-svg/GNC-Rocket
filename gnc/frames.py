"""Rotation / quaternion utilities.

Convention (ARCHITECTURE.md): scalar-first unit quaternion q = [w, x, y, z] that rotates body-frame vectors into
ENU, v_enu = R(q) @ v_b.  Body x points to the nose.  Hamilton product.  Functions taking q accept a single
quaternion (4,) or, where noted, a stack (..., 4).
"""
import numpy as np


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_normalize(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_conj(q):
    q = np.asarray(q, float)
    return q * np.array([1.0, -1, -1, -1])


def quat_mul(p, q):
    """Hamilton product p (x) q (supports stacks)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    pw, px, py, pz = np.moveaxis(p, -1, 0)
    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    return np.stack([pw * qw - px * qx - py * qy - pz * qz,
                     pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw], axis=-1)


def quat_to_dcm(q):
    """R(q) (body -> ENU), shape (..., 3, 3)."""
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def dcm_to_quat(R):
    """Inverse of quat_to_dcm for a single 3x3 matrix (Shepperd's method); returns q with w >= 0."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    i = int(np.argmax([tr, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = 2 * np.sqrt(1 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        s = 2 * np.sqrt(1 + 2 * R[0, 0] - tr)
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        s = 2 * np.sqrt(1 + 2 * R[1, 1] - tr)
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + 2 * R[2, 2] - tr)
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = quat_normalize(q)
    return q if q[0] >= 0 else -q


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_rotate(q, v):
    """Body -> ENU."""
    return quat_to_dcm(q) @ np.asarray(v, float)


def quat_rotate_inv(q, v):
    """ENU -> body."""
    return quat_to_dcm(q).T @ np.asarray(v, float)


def quat_derivative(q, omega_b):
    """q_dot = 0.5 q (x) [0, omega_b]  (omega in body frame)."""
    return 0.5 * quat_mul(q, np.concatenate([[0.0], np.asarray(omega_b, float)]))


def quat_to_euler_zyx(q):
    """3-2-1 Euler angles (yaw, pitch, roll) with R = Rz(yaw) Ry(pitch) Rx(roll) (ENU reference).
    Note body x = nose, so pitch = -(nose elevation above horizon). Supports stacks."""
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return yaw, pitch, roll


def euler_zyx_to_quat(yaw, pitch, roll):
    return quat_mul(quat_mul(quat_from_axis_angle([0, 0, 1], yaw), quat_from_axis_angle([0, 1, 0], pitch)),
                    quat_from_axis_angle([1, 0, 0], roll))


def nose_direction(q):
    """Body x axis in ENU (first column of R), supports stacks."""
    return quat_to_dcm(q)[..., :, 0]


def nose_elevation_azimuth(q):
    """(elevation above horizon, azimuth from North clockwise) of the nose, rad. Supports stacks."""
    n = nose_direction(q)
    return np.arcsin(np.clip(n[..., 2], -1, 1)), np.arctan2(n[..., 0], n[..., 1])


def tilt_from_vertical(q):
    """Angle between the nose and local Up, rad. Supports stacks."""
    return np.arccos(np.clip(nose_direction(q)[..., 2], -1, 1))


def launch_attitude(rail_angle_rad, rail_azimuth_rad):
    """Attitude on the rail: x_b along the rail (tilted rail_angle from vertical towards azimuth, from North
    clockwise), y_b = horizontal = Up x (azimuth direction) (i.e. North for an East launch), z_b = x_b x y_b
    (points up-range).  With this choice a positive body pitch rate q tilts the nose further down-range."""
    th, az = rail_angle_rad, rail_azimuth_rad
    h = np.array([np.sin(az), np.cos(az), 0.0])
    xb = np.cos(th) * np.array([0, 0, 1.0]) + np.sin(th) * h
    yb = np.cross([0, 0, 1.0], h)
    zb = np.cross(xb, yb)
    return dcm_to_quat(np.column_stack([xb, yb, zb]))
