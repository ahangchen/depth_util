import numpy as np
from numpy.linalg import inv


def depth2xyz(depth, cam, r_c2w, t_c2w, save_path):
    # for image warp
    h, w = depth.shape
    pix_coord = np.indices([h, w]).astype(np.float32)
    pix_coord = np.concatenate(
        (pix_coord, np.ones([1, h, w])), axis=0)
    pix_coord = np.reshape(pix_coord, [3, -1])
    depth = depth.reshape(-1)
    depth_3c = np.stack([depth, depth, depth]) # 3 * hxw
    t_c2w_rep = np.repeat(t_c2w, h * w).reshape(3, h * w)
    w_pts = r_c2w.dot(depth_3c * inv(cam).dot(pix_coord)) + t_c2w_rep # 3* hxw
    np.savetxt(save_path, w_pts.transpose(1, 0), fmt='%.6f', delimiter=' ')

if __name__ == '__main__':
    depth = np.ones((320, 256))
    cam = np.array([[480, 0, 160], [0, 480, 128], [0, 0, 1]])
    r_c2w = np.eye(3)
    t_c2w = np.arange(3)
    depth2xyz(depth, cam, r_c2w, t_c2w, '0.txt')
