import numpy as np
from network_s2d import Net
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import time
import os
from voxel_map import VoxelMap
from scipy import ndimage
from scipy.optimize import nnls
from dataclasses import dataclass
import matplotlib as mpl
import network_d2rpz
from PIL import Image


from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
device = torch.device("cpu")
np.set_printoptions(precision=3, suppress=True, linewidth=200)
torch.manual_seed(1)


######## TO BE CHANGED ###########

ABSOLUTE = False #True
EPOCHS = 50
VISU= True
CREATE_MOVIE = False #True

###########################

HEIGHTMAP_RES = 0.1
LABEL_NUMBER = 0
INPUT_PATH = './data/s2d_evaldata_rpz/'
OUTPUT_PATH = '/Users/zimmerk/python/rigid_body/output/10_kkt_slam/'

##### AUXILIARY FUNCTIONS #####
class Robot:
    def __init__(self, border_x, border_y, length, width, step, points,body,fr, fl, rr, lr,shapes):
        self.border_x = border_x
        self.border_y = border_y
        self.length = length
        self.width = width
        self.step = step
        self.points = points
        self.body =body
        self.rl = lr
        self.rr = rr
        self.fl = fl
        self.fr = fr
        self.shapes = shapes

    def transform_points(self, roll, pitch, yaw, x, y, z, points):
        Rx = np.zeros((3,3))
        Ry = np.zeros((3,3))
        Rz = np.zeros((3,3))
        t = np.zeros((3,1))

        Rx[0, 0] = 1
        Rx[1, 1] = np.cos(roll)
        Rx[1, 2] = -np.sin(roll)
        Rx[2, 1] = np.sin(roll)
        Rx[2, 2] = np.cos(roll)

        Ry[0, 0] = np.cos(pitch)
        Ry[0, 2] = np.sin(pitch)
        Ry[1, 1] = 1
        Ry[2, 0] = -np.sin(pitch)
        Ry[2, 2] = np.cos(pitch)

        Rz[0, 0] = np.cos(yaw)
        Rz[0, 1] = -np.sin(yaw)
        Rz[1, 0] = np.sin(yaw)
        Rz[1, 1] = np.cos(yaw)
        Rz[2, 2] = 1

        t[0, 0] = x
        t[1, 0] = y
        t[2, 0] = z

        return (np.matmul(np.matmul(np.matmul(Rx, Ry), Rz), points) + t.repeat( points.shape[1],1))  # transform pointcloud


    def plot_robot(self, ax , roll,pitch,yaw,x,y,z):
        points_body = self.transform_points(roll, pitch, yaw, x,y,z, self.body)
        points_fl_fr = self.transform_points(roll, pitch, yaw, x,y,z, self.fr)
        points_fl_fl = self.transform_points(roll, pitch, yaw, x,y,z, self.fl)
        points_fl_rr = self.transform_points(roll, pitch, yaw, x,y,z, self.rr)
        points_fl_rl = self.transform_points(roll, pitch, yaw, x,y,z,self.rl)
        self.plot_rotated_robot(ax, points_body, points_fl_fr, points_fl_fl, points_fl_rr, points_fl_rl, self.shapes)

    def plot_rotated_robot(self, ax, points_body, points_fl_fr, points_fl_fl, points_fl_rr, points_fl_rl, shapes):
        X = points_body[0].reshape(shapes[0])
        Y = points_body[1].reshape(shapes[0])
        Z = points_body[2].reshape(shapes[0])
        flipper_RF = []
        flipper_RF.append(points_fl_fr[0].reshape(shapes[1]))
        flipper_RF.append(points_fl_fr[1].reshape(shapes[1]))
        flipper_RF.append(points_fl_fr[2].reshape(shapes[1]))

        flipper_LF = []
        flipper_LF.append(points_fl_fl[0].reshape(shapes[2]))
        flipper_LF.append(points_fl_fl[1].reshape(shapes[2]))
        flipper_LF.append(points_fl_fl[2].reshape(shapes[2]))

        flipper_LB = []
        flipper_LB.append(points_fl_rl[0].reshape(shapes[3]))
        flipper_LB.append(points_fl_rl[1].reshape(shapes[3]))
        flipper_LB.append(points_fl_rl[2].reshape(shapes[3]))

        flipper_RB = []
        flipper_RB.append(points_fl_rr[0].reshape(shapes[4]))
        flipper_RB.append(points_fl_rr[1].reshape(shapes[4]))
        flipper_RB.append(points_fl_rr[2].reshape(shapes[4]))

        #ax = plt.axes(projection='3d')
        # PLOT TRACKS
        cltrack = 'k'
        clflipper = 'r'
        clbody = 'b'
        lw=2
        ax.plot_wireframe(X[0, :, :], Y[0, :, :], Z[0, :, :], color=cltrack, linewidth=lw)
        ax.plot_wireframe(X[1, :, :], Y[1, :, :], Z[1, :, :], color=cltrack, linewidth=lw)

        ax.plot_wireframe(X[-1, :, :], Y[-1, :, :], Z[-1, :, :], color=cltrack, linewidth=lw)
        ax.plot_wireframe(X[-2, :, :], Y[-2, :, :], Z[-2, :, :], color=cltrack, linewidth=lw)

        ax.plot_wireframe(X[0:2, :, 0], Y[0:2, :, 0], Z[0:2, :, 0], color=cltrack, linewidth=lw)
        ax.plot_wireframe(X[0:2, :, -1], Y[0:2, :, -1], Z[0:2, :, -1], color=cltrack, linewidth=lw)

        ax.plot_wireframe(X[-2:, :, 0], Y[-2:, :, 0], Z[-2:, :, 0], color=cltrack, linewidth=lw)
        ax.plot_wireframe(X[-2:, :, -1], Y[-2:, :, -1], Z[-2:, :, -1], color=cltrack, linewidth=lw)

        # PLOT BODY
        Z_body = np.concatenate([Z[1, :, 1][np.newaxis], Z[2:-2, :, 0], Z[-2, :, 1][np.newaxis]])
        ax.plot_wireframe(X[1:-1, :, 0], Y[1:-1, :, 0], Z_body, color=clbody, linewidth=lw)
        ax.plot_wireframe(X[1:-1, :, 0], Y[1:-1, :, 0], Z[1:-1, :, -1], color=clbody, linewidth=lw)

        # ax.plot_wireframe(X[-2,:,:], Y[-2,:,:],Z[-2,:,:], color=cltrack,linewidth=lw)
        # PLOT FLIPPER
        ax.plot_wireframe(flipper_LF[0][:, :, 0], flipper_LF[1][:, :, 0], flipper_LF[2][:, :, 0], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_LF[0][:, :, -1], flipper_LF[1][:, :, -1], flipper_LF[2][:, :, -1], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_LF[0][:, 0, :], flipper_LF[1][:, 0, :], flipper_LF[2][:, 0, :], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_LF[0][:, -1, :], flipper_LF[1][:, -1, :], flipper_LF[2][:, -1, :], color=clflipper,
                          linewidth=lw)
        # PLOT FLIPPER
        ax.plot_wireframe(flipper_RF[0][:, :, 0], flipper_RF[1][:, :, 0], flipper_RF[2][:, :, 0], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_RF[0][:, :, -1], flipper_RF[1][:, :, -1], flipper_RF[2][:, :, -1], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_RF[0][:, 0, :], flipper_RF[1][:, 0, :], flipper_RF[2][:, 0, :], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_RF[0][:, -1, :], flipper_RF[1][:, -1, :], flipper_RF[2][:, -1, :], color=clflipper,
                          linewidth=lw)
        # PLOT FLIPPER
        ax.plot_wireframe(flipper_RB[0][:, :, 0], flipper_RB[1][:, :, 0], flipper_RB[2][:, :, 0], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_RB[0][:, :, -1], flipper_RB[1][:, :, -1], flipper_RB[2][:, :, -1], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_RB[0][:, 0, :], flipper_RB[1][:, 0, :], flipper_RB[2][:, 0, :], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_RB[0][:, -1, :], flipper_RB[1][:, -1, :], flipper_RB[2][:, -1, :], color=clflipper,
                          linewidth=lw)
        # PLOT FLIPPER
        ax.plot_wireframe(flipper_LB[0][:, :, 0], flipper_LB[1][:, :, 0], flipper_LB[2][:, :, 0], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_LB[0][:, :, -1], flipper_LB[1][:, :, -1], flipper_LB[2][:, :, -1], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_LB[0][:, 0, :], flipper_LB[1][:, 0, :], flipper_LB[2][:, 0, :], color=clflipper,
                          linewidth=lw)
        ax.plot_wireframe(flipper_LB[0][:, -1, :], flipper_LB[1][:, -1, :], flipper_LB[2][:, -1, :], color=clflipper,
                          linewidth=lw)

        plt.show()

def robot_body_w_flippers(fl_flipper,fr_flipper,rl_flipper,rr_flipper):
    grid_res = 0.1

    ### Define rigid body (set of 3D points)
    length = 0.8
    width = 0.6
    robot_grid = np.meshgrid(np.arange(-length / 2, length / 2 + grid_res, grid_res),
                             np.arange(-width / 2, width / 2 + grid_res, grid_res),np.arange(-grid_res,grid_res,grid_res))
    ### Define tracks (set of 3D points)
    LT = np.zeros_like(robot_grid[0], dtype=int)
    LT[0:2, :,:] = 1
    RT = np.zeros_like(robot_grid[0], dtype=int)
    RT[5:, :,:] = 1
    X = robot_grid[0]
    Y = robot_grid[1]
    Z = robot_grid[2]
    Z[(LT == 0) & (RT == 0)] += 0.1
    N = Z.size
    P = np.stack((X.reshape([1, N]).squeeze(), Y.reshape([1, N]).squeeze(), Z.reshape([1, N]).squeeze()), 0)

    #ax.scatter(P[0, :], P[1, :], P[2, :], marker='x')

    flipper_length = 0.4
    flipper_width = 0.1
    flipper_LF = np.meshgrid(np.arange(0, flipper_length, grid_res / 2), np.arange(0, flipper_width, grid_res / 2), np.arange(-grid_res / 2, grid_res / 2 + grid_res, grid_res))
    flipper_LF[2][:, :, 1] -= np.ones_like(flipper_LF[2][:, :, 1]) * np.linspace(0, 0.02, flipper_LF[2][:, :, 1].shape[1])
    flipper_LF[2][:, :, 0] += np.ones_like(flipper_LF[2][:, :, 0]) * np.linspace(0, 0.02, flipper_LF[2][:, :, 0].shape[1])

    flipper_points = np.asarray([flipper_LF[0].ravel(),flipper_LF[1].ravel(),flipper_LF[2].ravel(),np.ones_like(flipper_LF[2].ravel())])
    angle = fl_flipper+np.pi
    R = np.asarray([[np.cos(angle),0,-np.sin(angle),0],
                    [0,1,0,0],
                   [np.sin(angle),0,np.cos(angle),0],
                   [0,0,0,1]])
    rot_flipper_points = np.matmul(R,flipper_points)
    flipper_LF[0] = np.reshape(rot_flipper_points[0],flipper_LF[0].shape)
    flipper_LF[1] = np.reshape(rot_flipper_points[1],flipper_LF[0].shape)
    flipper_LF[2] = np.reshape(rot_flipper_points[2],flipper_LF[0].shape)

    flipper_LF[0] += 0.3
    flipper_LF[1] += 0.3
    flipper_LF[2] -= 0.05


    flipper_RF = np.meshgrid(np.arange(0, flipper_length, grid_res / 2), np.arange(0, flipper_width, grid_res / 2), np.arange(-grid_res / 2, grid_res / 2 + grid_res, grid_res))
    flipper_points = np.asarray([flipper_LF[0].ravel(),flipper_LF[1].ravel(),flipper_LF[2].ravel(),np.ones_like(flipper_LF[2].ravel())])
    flipper_RF[2][:, :, 1] -= np.ones_like(flipper_RF[2][:, :, 1]) * np.linspace(0, 0.02, flipper_RF[2][:, :, 1].shape[1])
    flipper_RF[2][:, :, 0] += np.ones_like(flipper_RF[2][:, :, 0]) * np.linspace(0, 0.02, flipper_RF[2][:, :, 0].shape[1])
    flipper_points = np.asarray([flipper_RF[0].ravel(),flipper_RF[1].ravel(),flipper_RF[2].ravel(),np.ones_like(flipper_RF[2].ravel())])

    angle = fr_flipper+np.pi
    R = np.asarray([[np.cos(angle),0,-np.sin(angle),0],
                    [0,1,0,0],
                   [np.sin(angle),0,np.cos(angle),0],
                   [0,0,0,1]])
    rot_flipper_points = np.matmul(R,flipper_points)
    flipper_RF[0] = np.reshape(rot_flipper_points[0],flipper_RF[0].shape)
    flipper_RF[1] = np.reshape(rot_flipper_points[1],flipper_RF[0].shape)
    flipper_RF[2] = np.reshape(rot_flipper_points[2],flipper_RF[0].shape)

    flipper_RF[0] += 0.3
    flipper_RF[1] -= 0.35
    flipper_RF[2] -= 0.05


    flipper_RB = np.meshgrid(np.arange(0, flipper_length, grid_res / 2), np.arange(0, flipper_width, grid_res / 2), np.arange(-grid_res / 2, grid_res / 2 + grid_res, grid_res))
    flipper_RB[2][:, :, 1] -= np.ones_like(flipper_RB[2][:, :, 1]) * np.linspace(0, 0.02, flipper_RB[2][:, :, 1].shape[1])
    flipper_RB[2][:, :, 0] += np.ones_like(flipper_RB[2][:, :, 0]) * np.linspace(0, 0.02, flipper_RB[2][:, :, 0].shape[1])
    flipper_points = np.asarray([flipper_RB[0].ravel(),flipper_RB[1].ravel(),flipper_RB[2].ravel(),np.ones_like(flipper_RB[2].ravel())])
    angle = rr_flipper+np.pi/4
    R = np.asarray([[np.cos(angle),0,-np.sin(angle),0],
                    [0,1,0,0],
                   [np.sin(angle),0,np.cos(angle),0],
                   [0,0,0,1]])
    rot_flipper_points = np.matmul(R,flipper_points)
    flipper_RB[0] = np.reshape(rot_flipper_points[0],flipper_RB[0].shape)
    flipper_RB[1] = np.reshape(rot_flipper_points[1],flipper_RB[0].shape)
    flipper_RB[2] = np.reshape(rot_flipper_points[2],flipper_RB[0].shape)


    flipper_RB[0] -= 0.3
    flipper_RB[1] -= 0.35
    flipper_RB[2] -= 0.05


    flipper_LB = np.meshgrid(np.arange(0, flipper_length, grid_res / 2), np.arange(0, flipper_width, grid_res / 2), np.arange(-grid_res / 2, grid_res / 2 + grid_res, grid_res))
    flipper_LB[2][:, :, 1] -= np.ones_like(flipper_LB[2][:, :, 1]) * np.linspace(0, 0.02, flipper_LB[2][:, :, 1].shape[1])
    flipper_LB[2][:, :, 0] += np.ones_like(flipper_LB[2][:, :, 0]) * np.linspace(0, 0.02, flipper_LB[2][:, :, 0].shape[1])
    flipper_points = np.asarray([flipper_LB[0].ravel(),flipper_LB[1].ravel(),flipper_LB[2].ravel(),np.ones_like(flipper_LB[2].ravel())])
    angle = rl_flipper+np.pi/4
    R = np.asarray([[np.cos(angle),0,-np.sin(angle),0],
                    [0,1,0,0],
                   [np.sin(angle),0,np.cos(angle),0],
                   [0,0,0,1]])
    rot_flipper_points = np.matmul(R,flipper_points)
    flipper_LB[0] = np.reshape(rot_flipper_points[0],flipper_LB[0].shape)
    flipper_LB[1] = np.reshape(rot_flipper_points[1],flipper_LB[0].shape)
    flipper_LB[2] = np.reshape(rot_flipper_points[2],flipper_LB[0].shape)

    flipper_LB[0] -= 0.3
    flipper_LB[1] += 0.3
    flipper_LB[2] -= 0.05


    points_fl_fr = np.stack((flipper_RF[0].reshape(flipper_RF[0].size),flipper_RF[1].reshape(flipper_RF[0].size),flipper_RF[2].reshape(flipper_RF[0].size)),0)
    points_fl_fl = np.stack((flipper_LF[0].reshape(flipper_LF[0].size),flipper_LF[1].reshape(flipper_LF[0].size),flipper_LF[2].reshape(flipper_LF[0].size)),0)
    points_fl_rr = np.stack((flipper_RB[0].reshape(flipper_RB[0].size), flipper_RB[1].reshape(flipper_RB[0].size), flipper_RB[2].reshape(flipper_RB[0].size)), 0)
    #points_fl_rr =np.stack((flipper_RB[0].reshape(flipper_RB[0].size),flipper_RB[1].reshape(flipper_RB[0].size),flipper_LB[2].reshape(flipper_RB[0].size)),0)
    points_fl_rl = np.stack((flipper_LB[0].reshape(flipper_LB[0].size),flipper_LB[1].reshape(flipper_LB[0].size),flipper_LB[2].reshape(flipper_LB[0].size)),0)




    #points = np.hstack((P, RF, LF, RB, LB))
    points_body = P

    shapes = [robot_grid[0].shape, flipper_RF[0].shape, flipper_LF[0].shape, flipper_RB[0].shape, flipper_LB[0].shape]


    dx = -0.0
    points_body[0, :] = points_body[0, :] + dx
    points_fl_fr[0, :] = points_fl_fr[0, :] + dx
    points_fl_fl[0, :] = points_fl_fl[0, :] + dx
    points_fl_rr[0, :] = points_fl_rr[0, :] + dx
    points_fl_rl[0, :] = points_fl_rl[0, :] + dx

    dy = 0.0
    points_body[1, :] = points_body[1, :] + dy
    points_fl_fr[1, :] = points_fl_fr[1, :] + dy
    points_fl_fl[1, :] = points_fl_fl[1, :] + dy
    points_fl_rr[1, :] = points_fl_rr[1, :] + dy
    points_fl_rl[1, :] = points_fl_rl[1, :] + dy

    dz = 0.05
    points_body[2, :] = points_body[2, :] + dz
    points_fl_fr[2, :] = points_fl_fr[2, :] + dz
    points_fl_fl[2, :] = points_fl_fl[2, :] + dz
    points_fl_rr[2, :] = points_fl_rr[2, :] + dz
    points_fl_rl[2, :] = points_fl_rl[2, :] + dz


    return  points_body, points_fl_fr, points_fl_fl, points_fl_rr, points_fl_rl, shapes

def get_robot_model(length = 1.2, width = 0.6, grid_res = 0.1, fl_rl = 0, fl_rr = 0, fl_fr = 0, fl_fl = 0):

    points_body, points_fl_fr, points_fl_fl, points_fl_rr, points_fl_rl, shapes = robot_body_w_flippers(fl_fl, fl_fr, fl_rl, fl_rr)

    points =  np.hstack((points_body, points_fl_fr, points_fl_fl, points_fl_rr, points_fl_rl))
    #points = get_robot_model_simple(length=0.8, width=0.6, grid_res=0.1)

    robot = Robot(torch.tensor(length / 2), torch.tensor(width / 2), torch.tensor(length), torch.tensor(width), torch.tensor(grid_res), torch.tensor(points, dtype=torch.float32), points_body, points_fl_fr, points_fl_fl, points_fl_rr, points_fl_rl, shapes)

    return robot

def load_data(LABEL_NUMBER):
    ##### LOAD NETWORKS #######
    net = Net()
    net.load_state_dict(torch.load("./data/s2d_network/net_weights_s2d", map_location='cpu'))

    model_d2rpz = network_d2rpz.Net()
    model_d2rpz.load_state_dict(torch.load("./data/d2rpz_network/net_weights_d2rpz", map_location=device))
    model_d2rpz.to(device)


    ##### LOAD INPUT HEIGHTMAP #######
    l = LABEL_NUMBER
    data_path = './data/s2d_3d_palety_flippers'
    labels = os.listdir(data_path)
    labels.sort()
    label = np.load(data_path + '/' + labels[l])

    input = label['input'].astype(np.float32)
    input_mask = (~np.isnan(label['input'])).astype(np.float32)

    input[np.isnan(input)] = 0
    input = torch.tensor(input).unsqueeze(0).unsqueeze(0)
    input_mask = torch.tensor(input_mask).unsqueeze(0).unsqueeze(0)
    input_w_mask = torch.cat([input, input_mask], 1)

    ###### ESTIMATE DENSE DEM #########
    output_d = net(input_w_mask)
    output_d = output_d[0, 0, :, :].detach().cpu().numpy()

    ###### ESTIMATE RPZ SUBPSACE #########
    roll_all_rot = np.zeros([8, output_d.shape[0], output_d.shape[1]])
    pitch_all_rot = np.zeros([8, output_d.shape[0], output_d.shape[1]])
    z_all_rot = np.zeros([8, output_d.shape[0], output_d.shape[1]])

    r_pad = 3
    c_pad = 4
    yaw_i = 0
    for yaw_rot in np.linspace(0, 315, 8).tolist():
        dem_rotate = np.asarray(Image.fromarray(output_d[:, :]).rotate(yaw_rot))
        input_dem = torch.from_numpy(dem_rotate).unsqueeze(0).unsqueeze(0)
        output_d2rpz = model_d2rpz(input_dem[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3]).detach()  # self.model_d2rpz(input_dem)
        output_d2rpz = output_d2rpz.detach().cpu().numpy()
        roll_d2rpz_grid = np.asarray(Image.fromarray(output_d2rpz[0, 0, :, :]).rotate(-yaw_rot))
        pitch_d2rpz_grid = np.asarray(Image.fromarray(output_d2rpz[1, 0, :, :]).rotate(-yaw_rot))
        z_d2rpz_grid = np.asarray(Image.fromarray(output_d2rpz[2, 0, :, :]).rotate(-yaw_rot))

        roll_all_rot[yaw_i, r_pad:-r_pad, c_pad:-c_pad] = roll_d2rpz_grid[:, :]
        pitch_all_rot[yaw_i, r_pad:-r_pad, c_pad:-c_pad] = pitch_d2rpz_grid[:, :]
        z_all_rot[yaw_i, r_pad:-r_pad, c_pad:-c_pad] = z_d2rpz_grid[:, :]

        yaw_i += 1

    return output_d, torch.tensor(roll_all_rot), torch.tensor(pitch_all_rot), torch.tensor(z_all_rot)


##### MAIN FUNCTIONS #####

class Rpz_diff:
    def __init__(self, sz = 5 ):
        self.sz = sz
        self.kernel_sz = 2 * sz + 1
        sig = self.kernel_sz / 5
        sx = np.linspace(-(self.kernel_sz - 1) / 2., (self.kernel_sz - 1) / 2., self.kernel_sz)
        xx, yy = np.meshgrid(sx, sx)
        self.xx = torch.tensor(xx)
        self.yy = torch.tensor(yy)
        self.sig2 = torch.tensor(sig).square()
        #self.sig2.requires_grad_(True)  # could be directly optimized as well

    def interp_yaw(self, yaw_in):
        yaw_deg = np.degrees(yaw_in.detach().numpy().squeeze())
        yaw_bins = np.linspace(0, 315 + 45, 9)
        if yaw_deg >= 360:
            yaw_deg = yaw_deg - 360
        if yaw_deg < 0:
            yaw_deg = yaw_deg + 360
        idx = np.digitize(yaw_deg, yaw_bins)
        if 0 < idx < 8:
            yaw_idx = [idx - 1, idx]
        if idx == 8:
            yaw_idx = [idx - 1, 0]
        w = (np.radians(yaw_bins[yaw_idx[1]]) - yaw_in) / (np.pi/4)
        return yaw_idx, w

    def t2rpyxyz(self, t, x0, y0, roll_all, pitch_all, z_all, sig2=None):
        #  vezme trajektorii (t,x0,y0) a vrati diferenctovatelne (roll, pitch, yaw, x, y, z)
        if sig2 is None:
            sig2 = self.sig2

        TRAJECTORY_LENGTH = t.shape[0]+1

        roll = torch.zeros([TRAJECTORY_LENGTH, 1])
        pitch = torch.zeros([TRAJECTORY_LENGTH, 1])
        yaw = torch.zeros([TRAJECTORY_LENGTH, 1])
        x = torch.zeros([TRAJECTORY_LENGTH, 1])
        y = torch.zeros([TRAJECTORY_LENGTH, 1])
        z = torch.zeros([TRAJECTORY_LENGTH, 1])

        for k in range(TRAJECTORY_LENGTH):
            if k==0:
                x[k] = x0
                y[k] = y0
                yaw[k] = 0
            else:
                yaw[k] = t[k - 1, 1]
                x[k] = x[k - 1] + t[k - 1, 0] * torch.cos(t[k - 1, 1])
                y[k] = y[k - 1] + t[k - 1, 0] * torch.sin(t[k - 1, 1])
                yaw_idx, w = self.interp_yaw(yaw[k])


                C = x[k].long()
                R = y[k].long()
                kernel = torch.exp(-0.5 * ((self.xx + C - x[k]).square() + (self.yy + R - y[k]).square()) / sig2)
                kernel = kernel/kernel.sum()

                roll[k] = (kernel * (roll_all[yaw_idx[0], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * w + roll_all[yaw_idx[1], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * (1 - w))).sum()
                pitch[k] = (kernel * (pitch_all[yaw_idx[0], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * w + pitch_all[yaw_idx[1], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * (1 - w))).sum()
                z[k] = (kernel * (z_all[yaw_idx[0], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * w + z_all[yaw_idx[1], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * (1 - w))).sum()
                ROLL[k] = roll_all[0, R, C]
                PITCH[k] = pitch_all[0, R, C]
                Z[k] = z_all[0, R, C]

        return roll, pitch, yaw, x, y, z, ROLL, PITCH, Z


    def t2xyy(self, t, x0, y0):
        #  vezme relativni trajektorii (t,x0,y0) a vrati absolutni (x, y, yaw)
        yaw = torch.zeros([TRAJECTORY_LENGTH, 1])
        x = torch.zeros([TRAJECTORY_LENGTH, 1])
        y = torch.zeros([TRAJECTORY_LENGTH, 1])

        for k in range(TRAJECTORY_LENGTH):
            if k == 0:
                x[k] = x0
                y[k] = y0
                yaw[k] = 0
            else:
                yaw[k] = t[k - 1, 1]
                x[k] = x[k - 1] + t[k - 1, 0] * torch.cos(t[k - 1, 1])
                y[k] = y[k - 1] + t[k - 1, 0] * torch.sin(t[k - 1, 1])
        return x,y,yaw

    def xy2t(self, x, y):
        #  vezme absolutni trajektorii (x,y) a vrati relativni trajektorii (t, x0, y0)
        t = torch.zeros([TRAJECTORY_LENGTH-1, 2])
        for k in range(TRAJECTORY_LENGTH):
            if k == 0:
                x0 = x[k].squeeze()
                y0 = y[k].squeeze()
            else:
                t[k-1, 0] = torch.sqrt((x[k] - x[k - 1]).square() + (y[k] - y[k - 1]).square())  # velocity
                t[k-1, 1] = torch.atan2(y[k] - y[k-1], x[k] - x[k-1]) # yaw
        return t, x0, y0

    def xyy2rpz(self, x, y, yaw, roll_all, pitch_all, z_all, sig2=None):
        #  vezme absolutni trajektorii (x, y, yaw) a vrati diferencovatelne (roll, pitch, z)
        if sig2 is None:
            sig2 = self.sig2

        TRAJECTORY_LENGTH = t.shape[0]+1

        roll = torch.zeros([TRAJECTORY_LENGTH, 1])
        pitch = torch.zeros([TRAJECTORY_LENGTH, 1])
        z = torch.zeros([TRAJECTORY_LENGTH, 1])

        for k in range(TRAJECTORY_LENGTH):
                yaw_idx, w = self.interp_yaw(yaw[k])
                C = x[k].long()
                R = y[k].long()
                kernel = torch.exp(-0.5 * ((self.xx + C - x[k]).square() + (self.yy + R - y[k]).square()) / sig2)
                kernel = kernel / kernel.sum()

                roll[k] = (kernel * (roll_all[yaw_idx[0], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * w + roll_all[yaw_idx[1], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * (1 - w))).sum()
                pitch[k] = (kernel * (pitch_all[yaw_idx[0], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * w + pitch_all[yaw_idx[1], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * (1 - w))).sum()
                z[k] = (kernel * (z_all[yaw_idx[0], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * w + z_all[yaw_idx[1], (R - self.sz):(R + self.sz + 1), (C - self.sz):(C + self.sz + 1)] * (1 - w))).sum()
                ROLL[k] = roll_all[0, R, C]
                PITCH[k] = pitch_all[0, R, C]
                Z[k] = z_all[0, R, C]

        return roll, pitch, z, ROLL, PITCH, Z

def visualize(x,  robot, output_dem, LOSS, az = 0):
    plt.figure(2)
    plt.clf()
    ax = plt.axes(projection='3d')
    Z = output_dem
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    XX = X * HEIGHTMAP_RES
    YY = Y * HEIGHTMAP_RES

    surf = ax.plot_surface(XX, YY, Z, cmap = 'jet', linewidth=0, antialiased=False, alpha=0.4)
    plt.title('LOSS = ' + str('{0:.2f}'.format(LOSS.squeeze().detach().numpy())) )

    for k in range(x.shape[0]):
        #if (np.divmod(k, 5)[1] == 0):
        robot.plot_robot(ax, x[k,0].detach().numpy(), x[k,1].detach().numpy(), x[k,2].detach().numpy(), x[k,3].detach().numpy() * HEIGHTMAP_RES, x[k, 4].detach().numpy() * HEIGHTMAP_RES, x[k, 5].detach().numpy())

    #ax.scatter(pcl[:, 0].detach().numpy(), pcl[:, 1].detach().numpy(), pcl[:, 2].detach().numpy(), color = 'red')

    plt.xlabel('x - cols')
    plt.ylabel('y - rows')
    ax.set_xlim3d(4, 12)
    ax.set_ylim3d(8, 16)
    #ax.set_zlim3d(-0.2, 0.4)
    ax.view_init(elev=80, azim=az)  # elev=15, azim=visu)

    plt.pause(0.001)
    plt.draw()

def visualize_simple(x, y, output_dem):
    plt.figure(1)
    plt.clf()
    plt.imshow((output_dem), cmap='jet', origin='lower')
    plt.plot(x.detach().numpy(),y.detach().numpy(),'o', linewidth = 3, markersize = 10, color = 'w')
    plt.plot(x.detach().numpy(),y.detach().numpy(),'o', linewidth = 3, markersize = 7, color = 'b')
    plt.title('LOSS = ' + str('{0:.2f}'.format(LOSS.squeeze().detach().numpy())) )
    plt.xlabel('x - cols')
    plt.ylabel('y - rows')
    plt.pause(0.001)
    plt.draw()


############################
########  MAIN CODE  #######
############################


##### Initialize ROBOT (just for visualization purposes) #####
robot = get_robot_model(length=1.2, width=0.6, grid_res=HEIGHTMAP_RES)

##### Load RPZ-subspace #####
output_dem, roll_all, pitch_all, z_all = load_data(LABEL_NUMBER)

##### Initialize Rpz_diff #####
rpz = Rpz_diff(1)

##### Define initial trajectory #####
TRAJECTORY_LENGTH = 10
x = 60 + 10*torch.arange(0, TRAJECTORY_LENGTH, dtype=float).view(TRAJECTORY_LENGTH, 1)
y = 100 + 0*torch.arange(0, TRAJECTORY_LENGTH, dtype=float).view(TRAJECTORY_LENGTH, 1)
yaw = torch.zeros([TRAJECTORY_LENGTH, 1])

if ~ABSOLUTE:
    t, x0, y0 = rpz.xy2t(x.detach(), y.detach())
    # you can go back to RELATIVE represenation by
    # x,y,yaw = rpz.t2xyy(t, x0, y0)

################################## OPTIMIZE TRAJECTORY ON RPZ SUBSPACE ##################################

if ABSOLUTE:
    K_v = 0.001
    K_yaw = 0.1
    optimizer = optim.Adam([x, y, yaw], lr=0.1)
    x0 = x.clone()
    y0 = y.clone()
    yaw0 = yaw.clone()
    x.requires_grad_(True)
    y.requires_grad_(True)
    yaw.requires_grad_(True)
else:
    K_v = 0.1
    K_yaw = 0.1
    optimizer = optim.Adam([t], lr=0.1)
    t0 = t.clone()
    t.requires_grad_(True)


ROLL = torch.zeros([TRAJECTORY_LENGTH, 1])
PITCH = torch.zeros([TRAJECTORY_LENGTH, 1])
Z = torch.zeros([TRAJECTORY_LENGTH, 1])


for epoch in range(EPOCHS):
    print("epoch:", epoch)

    # zero the parameter gradients
    optimizer.zero_grad()

    # define loss
    time_start = time.time()

    if ABSOLUTE:
        roll, pitch, z, ROLL, PITCH, Z = rpz.xyy2rpz(x, y, yaw, roll_all, pitch_all, z_all)
    else:
        roll, pitch, yaw, x, y, z, ROLL, PITCH, Z = rpz.t2rpyxyz(t, x0, y0, roll_all, pitch_all, z_all)


    print("rpz_diff time:", time.time() - time_start, "sec")

    if ABSOLUTE:
        LOSS = (z.square().sum() + pitch.square().sum() + roll.square().sum() )+ K_v * (x - x0).square().sum() + K_v * (y - y0).square().sum() + K_yaw * (yaw - yaw0).square().sum()
    else:
        LOSS = (z.square().sum() + pitch.square().sum() + roll.square().sum() )+ K_v * (t[:,0] - t0[:,0]).square().sum() + K_yaw * (t[:,1] - t0[:,1]).square().sum()

    #visualize(torch.cat((ROLL, PITCH, yaw, x, y, Z), dim=1), robot, output_dem, LOSS, -90 )

    visualize_simple(x, y, output_dem)

    # estimate gradient
    time_start = time.time()
    LOSS.backward()
    print("LOSS.backward() time:", time.time() - time_start, "sec")
    print('loss:', LOSS.detach().numpy() )


    # update trajectory
    optimizer.step()

    if CREATE_MOVIE:
            plt.savefig(OUTPUT_PATH + '{:04d}_kkt_slam'.format(epoch) + '.png')

if CREATE_MOVIE:
    os.system('rm ' + OUTPUT_PATH + 'output.mp4')
    os.system('ffmpeg -i ' + OUTPUT_PATH + '%04d_kkt_slam.png -c:v libx264 -vf scale=1280:-2 -pix_fmt yuv420p ' + OUTPUT_PATH + 'output.mp4')

######################################################################################
######################################################################################
######################################################################################
