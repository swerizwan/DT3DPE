import numpy as np

# Fast cross product implementation
def _fast_cross(a, b):
    """
    Computes the cross product of two vectors.
    :param a: First vector.
    :param b: Second vector.
    :return: Cross product of a and b.
    """
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

# Creates an identity quaternion
def eye(shape, dtype=np.float32):
    """
    Creates an identity quaternion with the given shape.
    :param shape: Shape of the quaternion.
    :param dtype: Data type of the quaternion.
    :return: Identity quaternion.
    """
    return np.ones(list(shape) + [4], dtype=dtype) * np.asarray([1, 0, 0, 0], dtype=dtype)

# Computes the length of a vector
def length(x):
    """
    Computes the length (magnitude) of a vector.
    :param x: Input vector.
    :return: Length of the vector.
    """
    return np.sqrt(np.sum(x * x, axis=-1))

# Normalizes a vector
def normalize(x, eps=1e-8):
    """
    Normalizes a vector.
    :param x: Input vector.
    :param eps: Epsilon value to avoid division by zero.
    :return: Normalized vector.
    """
    return x / (length(x)[...,None] + eps)

# Computes the absolute value of a quaternion
def abs(x):
    """
    Computes the absolute value of a quaternion.
    :param x: Input quaternion.
    :return: Absolute value of the quaternion.
    """
    return np.where(x[...,0:1] > 0.0, x, -x)

# Inverts a quaternion
def inv(q):
    """
    Inverts a quaternion.
    :param q: Input quaternion.
    :return: Inverted quaternion.
    """
    return np.array([1, -1, -1, -1], dtype=np.float32) * q

# Computes the dot product of two vectors
def dot(x, y):
    """
    Computes the dot product of two vectors.
    :param x: First vector.
    :param y: Second vector.
    :return: Dot product of x and y.
    """
    return np.sum(x * y, axis=-1)[...,None] if x.ndim > 1 else np.sum(x * y, axis=-1)

# Multiplies two quaternions
def mul(x, y):
    """
    Multiplies two quaternions.
    :param x: First quaternion.
    :param y: Second quaternion.
    :return: Product of x and y.
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

# Multiplies a quaternion with the inverse of another quaternion
def inv_mul(x, y):
    """
    Multiplies a quaternion with the inverse of another quaternion.
    :param x: First quaternion.
    :param y: Second quaternion.
    :return: Product of x and the inverse of y.
    """
    return mul(inv(x), y)

# Multiplies a vector with a quaternion
def mul_vec(q, x):
    """
    Multiplies a vector with a quaternion.
    :param q: Quaternion.
    :param x: Vector.
    :return: Result of the multiplication.
    """
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., None] * t + _fast_cross(q[..., 1:], t)

# Multiplies a vector with the inverse of a quaternion
def inv_mul_vec(q, x):
    """
    Multiplies a vector with the inverse of a quaternion.
    :param q: Quaternion.
    :param x: Vector.
    :return: Result of the multiplication.
    """
    return mul_vec(inv(q), x)

# Unrolls a quaternion sequence to avoid discontinuities
def unroll(x):
    """
    Unrolls a quaternion sequence to avoid discontinuities.
    :param x: Quaternion sequence.
    :return: Unrolled quaternion sequence.
    """
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum( y[i] * y[i-1], axis=-1)
        d1 = np.sum(-y[i] * y[i-1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y

# Computes the quaternion between two vectors
def between(x, y):
    """
    Computes the quaternion between two vectors.
    :param x: First vector.
    :param y: Second vector.
    :return: Quaternion between x and y.
    """
    return np.concatenate([
        np.sqrt(np.sum(x*x, axis=-1) * np.sum(y*y, axis=-1))[...,None] + 
        np.sum(x * y, axis=-1)[...,None], 
        _fast_cross(x, y)], axis=-1)
        
# Computes the logarithm of a quaternion
def log(x, eps=1e-5):
    """
    Computes the logarithm of a quaternion.
    :param x: Input quaternion.
    :param eps: Epsilon value to avoid division by zero.
    :return: Logarithm of the quaternion.
    """
    length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,None]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / length)
    return halfangle * x[...,1:]

# Computes the exponential of a quaternion
def exp(x, eps=1e-5):
    """
    Computes the exponential of a quaternion.
    :param x: Input quaternion.
    :param eps: Epsilon value to avoid division by zero.
    :return: Exponential of the quaternion.
    """
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[...,None]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)

# Forward kinematics for rotations and positions
def fk(lrot, lpos, parents):
    """
    Forward kinematics for rotations and positions.
    :param lrot: Local rotations.
    :param lpos: Local positions.
    :param parents: Parent indices.
    :return: Global rotations and positions.
    """
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[...,i:i+1,:]))
        
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)

# Forward kinematics for rotations only
def fk_rot(lrot, parents):
    """
    Forward kinematics for rotations only.
    :param lrot: Local rotations.
    :param parents: Parent indices.
    :return: Global rotations.
    """
    gr = [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gr.append(mul(gr[parents[i]], lrot[...,i:i+1,:]))
        
    return np.concatenate(gr, axis=-2)

# Inverse kinematics for rotations and positions
def ik(grot, gpos, parents):
    """
    Inverse kinematics for rotations and positions.
    :param grot: Global rotations.
    :param gpos: Global positions.
    :param parents: Parent indices.
    :return: Local rotations and positions.
    """
    return (
        np.concatenate([
            grot[...,:1,:],
            mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
        ], axis=-2),
        np.concatenate([
            gpos[...,:1,:],
            mul_vec(
                inv(grot[...,parents[1:],:]),
                gpos[...,1:,:] - gpos[...,parents[1:],:]),
        ], axis=-2))

# Inverse kinematics for rotations only
def ik_rot(grot, parents):
    """
    Inverse kinematics for rotations only.
    :param grot: Global rotations.
    :param parents: Parent indices.
    :return: Local rotations.
    """
    return np.concatenate([grot[...,:1,:], 
                        mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
                    ], axis=-2)
    
# Forward kinematics for velocities and angular velocities
def fk_vel(lrot, lpos, lvel, lang, parents):
    """
    Forward kinematics for velocities and angular velocities.
    :param lrot: Local rotations.
    :param lpos: Local positions.
    :param lvel: Local velocities.
    :param lang: Local angular velocities.
    :param parents: Parent indices.
    :return: Global rotations, positions, velocities, and angular velocities.
    """
    gp, gr, gv, ga = [lpos[...,:1,:]], [lrot[...,:1,:]], [lvel[...,:1,:]], [lang[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[...,i:i+1,:]))
        gv.append(mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) + 
            _fast_cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[...,i:i+1,:])) +
            gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[...,i:i+1,:]) + ga[parents[i]])
        
    return (
        np.concatenate(gr, axis=-2), 
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))

# Linear interpolation
def lerp(x, y, t):
    """
    Linear interpolation between two values.
    :param x: First value.
    :param y: Second value.
    :param t: Interpolation factor.
    :return: Interpolated value.
    """
    return (1 - t) * x + t * y

# Quaternion linear interpolation
def quat_lerp(x, y, t):
    """
    Quaternion linear interpolation.
    :param x: First quaternion.
    :param y: Second quaternion.
    :param t: Interpolation factor.
    :return: Interpolated quaternion.
    """
    return normalize(lerp(x, y, t))

# Spherical linear interpolation
def slerp(x, y, t):
    """
    Spherical linear interpolation.
    :param x: First quaternion.
    :param y: Second quaternion.
    :param t: Interpolation factor.
    :return: Interpolated quaternion.
    """
    if t == 0:
        return x
    elif t == 1:
        return y
    
    if dot(x, y) < 0:
        y = - y
    ca = dot(x, y)
    theta = np.arccos(np.clip(ca, 0, 1))
    
    r = normalize(y - x * ca)
    
    return x * np.cos(theta * t) + r * np.sin(theta * t)

# Converts a quaternion to Euler angles
def to_euler(x, order='zyx'):
    """
    Converts a quaternion to Euler angles.
    :param x: Quaternion.
    :param order: Order of the Euler angles.
    :return: Euler angles.
    """
    q0 = x[...,0:1]
    q1 = x[...,1:2]
    q2 = x[...,2:3]
    q3 = x[...,3:4]
    
    if order == 'zyx':
        return np.concatenate([
            np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)),
            np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1,1)),
            np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))], axis=-1)
    elif order == 'yzx':
        return np.concatenate([
            np.arctan2(2 * (q2 * q0 - q1 * q3),  q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0),
            np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1,1)), 
            np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)],axis=-1)
    elif order == 'zxy':
        return np.concatenate([
            np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3),
            np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1,1)),
            np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)], axis=-1)
    elif order == 'yxz':
        return np.concatenate([
            np.arctan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3),
            np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1,1)),
            np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)], axis=-1)
    else:
        raise NotImplementedError('Cannot convert from ordering %s' % order)

# Converts a quaternion to a transformation matrix
def to_xform(x):
    """
    Converts a quaternion to a transformation matrix.
    :param x: Quaternion.
    :return: Transformation matrix.
    """
    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[...,None,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[...,None,:],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[...,None,:],
    ], axis=-2)

# Converts a quaternion to a 2D transformation matrix
def to_xform_xy(x):
    """
    Converts a quaternion to a 2D transformation matrix.
    :param x: Quaternion.
    :return: 2D transformation matrix.
    """
    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz], axis=-1)[...,None,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz)], axis=-1)[...,None,:],
        np.concatenate([xz - wy, yz + wx], axis=-1)[...,None,:],
    ], axis=-2)

# Converts a quaternion to a scaled angle-axis representation
def to_scaled_angle_axis(x, eps=1e-5):
    """
    Converts a quaternion to a scaled angle-axis representation.
    :param x: Quaternion.
    :param eps: Epsilon value to avoid division by zero.
    :return: Scaled angle-axis representation.
    """
    return 2.0 * log(x, eps)

# Converts an angle and axis to a quaternion
def from_angle_axis(angle, axis):
    """
    Converts an angle and axis to a quaternion.
    :param angle: Angle.
    :param axis: Axis.
    :return: Quaternion.
    """
    c = np.cos(angle / 2.0)[..., None]
    s = np.sin(angle / 2.0)[..., None]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

# Converts an axis-angle representation to a quaternion
def from_axis_angle(rots):
    """
    Converts an axis-angle representation to a quaternion.
    :param rots: Axis-angle representation.
    :return: Quaternion.
    """
    angle = np.linalg.norm(rots, axis=-1)
    axis = rots / angle[...,None]
    return from_angle_axis(angle, axis)

# Converts Euler angles to a quaternion
def from_euler(e, order='zyx'):
    """
    Converts Euler angles to a quaternion.
    :param e: Euler angles.
    :param order: Order of the Euler angles.
    :return: Quaternion.
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))

# Converts a transformation matrix to a quaternion
def from_xform(ts):
    """
    Converts a transformation matrix to a quaternion.
    :param ts: Transformation matrix.
    :return: Quaternion.
    """
    return normalize(
        np.where((ts[...,2,2] < 0.0)[...,None],
            np.where((ts[...,0,0] >  ts[...,1,1])[...,None],
                np.concatenate([
                    (ts[...,2,1]-ts[...,1,2])[...,None], 
                    (1.0 + ts[...,0,0] - ts[...,1,1] - ts[...,2,2])[...,None], 
                    (ts[...,1,0]+ts[...,0,1])[...,None], 
                    (ts[...,0,2]+ts[...,2,0])[...,None]], axis=-1),
                np.concatenate([
                    (ts[...,0,2]-ts[...,2,0])[...,None], 
                    (ts[...,1,0]+ts[...,0,1])[...,None], 
                    (1.0 - ts[...,0,0] + ts[...,1,1] - ts[...,2,2])[...,None], 
                    (ts[...,2,1]+ts[...,1,2])[...,None]], axis=-1)),
            np.where((ts[...,0,0] < -ts[...,1,1])[...,None],
                np.concatenate([
                    (ts[...,1,0]-ts[...,0,1])[...,None], 
                    (ts[...,0,2]+ts[...,2,0])[...,None], 
                    (ts[...,2,1]+ts[...,1,2])[...,None], 
                    (1.0 - ts[...,0,0] - ts[...,1,1] + ts[...,2,2])[...,None]], axis=-1),
                np.concatenate([
                    (1.0 + ts[...,0,0] + ts[...,1,1] + ts[...,2,2])[...,None], 
                    (ts[...,2,1]-ts[...,1,2])[...,None], 
                    (ts[...,0,2]-ts[...,2,0])[...,None], 
                    (ts[...,1,0]-ts[...,0,1])[...,None]], axis=-1))))

# Converts a 2D transformation matrix to a quaternion
def from_xform_xy(x):
    """
    Converts a 2D transformation matrix to a quaternion.
    :param x: 2D transformation matrix.
    :return: Quaternion.
    """
    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / np.sqrt(np.sum(np.square(c2), axis=-1))[...,None]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / np.sqrt(np.sum(np.square(c1), axis=-1))[...,None]
    c0 = x[...,0]
    
    return from_xform(np.concatenate([
        c0[...,None], 
        c1[...,None], 
        c2[...,None]], axis=-1))

# Converts a scaled angle-axis representation to a quaternion
def from_scaled_angle_axis(x, eps=1e-5):
    """
    Converts a scaled angle-axis representation to a quaternion.
    :param x: Scaled angle-axis representation.
    :param eps: Epsilon value to avoid division by zero.
    :return: Quaternion.
    """
    return exp(x / 2.0, eps)