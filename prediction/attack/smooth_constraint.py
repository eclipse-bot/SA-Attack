import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
import torch

dl = 0.05
clothoid_num = 12

# Vehicle parameters (m)
LENGTH = 4.5
WIDTH = 2.0
BACKTOWHEEL = 1.0
WHEEL_LEN = 0.3
WHEEL_WIDTH = 0.2
TREAD = 0.7
WB = 2.5


def plotVehicle(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
    """
    The function is to plot the vehicle
    it is copied from https://github.com/AtsushiSakai/PythonRobotics/blob/187b6aa35f3cbdeca587c0abdb177adddefc5c2a/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py#L109
    """
    outline = np.array(
        [
            [
                -BACKTOWHEEL,
                (LENGTH - BACKTOWHEEL),
                (LENGTH - BACKTOWHEEL),
                -BACKTOWHEEL,
                -BACKTOWHEEL,
            ],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
        ]
    )

    fr_wheel = np.array(
        [
            [WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
            [
                -WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
            ],
        ]
    )

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array(
        [[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]]
    )

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(
        np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), truckcolor
    )
    plt.plot(
        np.array(fr_wheel[0, :]).flatten(),
        np.array(fr_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(rr_wheel[0, :]).flatten(),
        np.array(rr_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(fl_wheel[0, :]).flatten(),
        np.array(fl_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(rl_wheel[0, :]).flatten(),
        np.array(rl_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(x, y, "*")

def getDistance(p1, p2):
    """
    Calculate distance
    :param p1: list, point1
    :param p2: list, point2
    :return: float, distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

def rotate_delta(point, delta):
    """
    Calculate point rotate delta
    :param point:
    :param delta:
    :return: point
    """
    result = []
    result.append(point[0] * math.cos(delta) + point[1] * math.sin(delta))
    result.append((-1) * point[0] * math.sin(delta) + point[1] * math.cos(delta))
    return result
#..
class Vehicle:
    def __init__(self, x, y, yaw, curva=0, vel=0):
        """
        Define a vehicle class
        :param x: float, x position
        :param y: float, y position
        :param yaw: float, vehicle heading
        :param vel: float, velocity
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.curva = curva
        self.vel = vel
    def update(self, d_curva):
        """
        Vehicle motion model, here we are using kinematic bycicle model
        :param acc: float, acceleration
        :param curva: float, curvature
        """
        self.x += math.cos(self.yaw) * dl
        self.y += math.sin(self.yaw) * dl
        self.yaw += self.curva * dl
        self.curva += d_curva * dl
        #self.vel += acc * dt


def insert_generation(start_point, end_point, num):
    """
    :param start_point:[x0, y0]
    :param end_point: [xt, yt]
    :param num: int number
    :return:list x, list y
    """
    distance = np.linalg.norm(end_point - start_point)
    interval = distance / (num + 1)
    points = np.zeros((num + 2, 2))
    points[0] = start_point
    points[-1] = end_point
    for i in range(1, num + 1):
        points[i] = start_point + (i * interval) * (end_point - start_point) / distance
    return points


class Trajectory:
    def __init__(self, traj_x, traj_y, L):
        """
        Define a trajectory class
        :param traj_x: list, list of x position
        :param traj_y: list, list of y position
        """
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.L = L
        self.last_idx = 0

    def getPoint(self, idx):
        return [self.traj_x[idx], self.traj_y[idx]]

    def getTargetPoint(self, pos):
        """
        Get the next look ahead point
        :param pos: list, vehicle position
        :return: list, target point
        """
        screen_list = []
        for l in range(len(self.traj_x)):
            point = self.getPoint(l)
            if getDistance(pos,point) < self.L:
                screen_list.append(point)
        if screen_list:
            target_point = screen_list[-1]
            #print(len(screen_list))
        else:
            print("车辆偏移超出追踪距离")#better process
            sys.exit()
        return target_point

def smooth_constraint(data, obj, perturbation, attack_length, look_distance, insert_num):

    if not isinstance(perturbation, np.ndarray):
        perturbation_array = perturbation.cpu().detach().numpy()
    else:
        perturbation_array = perturbation
    total_traj = copy.deepcopy(data["objects"][obj]["observe_trace"])
    heading = copy.deepcopy(data["objects"][obj]["observe_feature"][:,4])
    a=total_traj[1]-total_traj[0]

    if a[0] <0:
        b=math.atan((a[1] / a[0]))+math.pi
    else:
        b=math.atan((a[1] / a[0]))
    ego = Vehicle(total_traj[0][0], total_traj[0][1], heading[0])
    total_traj[:attack_length] = total_traj[:attack_length] + perturbation_array

    #insert data
    for i in range(attack_length):
        ins_start = total_traj[i]
        ins_end = total_traj[i+1]
        trajectory = insert_generation(ins_start, ins_end, insert_num)
        trajectory = np.delete(trajectory, 0, axis=0)
        if i==0:
            traj_x = trajectory[:,0]
            traj_y = trajectory[:,1]
        else:
            traj_x = np.concatenate((traj_x, trajectory[:,0]),axis = 0)
            traj_y = np.concatenate((traj_y, trajectory[:,1]),axis = 0)
    traj = Trajectory(traj_x, traj_y, look_distance)
    goal = traj.getPoint(len(traj_x) - 1)

    # real trajectory
    traj_ego_x = []
    traj_ego_y = []

    #pure_pursuit
    #plt.figure(figsize=(12, 8))
    while getDistance([ego.x, ego.y], goal) > 0.5:
        target_point = traj.getTargetPoint([ego.x, ego.y])
        rel_target_point = [target_point[0] - ego.x, target_point[1] - ego.y]
        desired_curva = 2 * (rotate_delta(rel_target_point, ego.yaw)[1]) / \
                        (getDistance([ego.x, ego.y], target_point) * getDistance([ego.x, ego.y],
                                                                                 target_point))
        d_curva = (desired_curva - ego.curva) / (clothoid_num * dl)
        # print("d_curva：", d_curva)
        for clothoid_id in range(clothoid_num):
            # move the vehicle
            ego.update(d_curva)
            # store the trajectory
            traj_ego_x.append(ego.x)
            traj_ego_y.append(ego.y)

        # # plots
        # plt.cla()
        # plt.plot(traj_x, traj_y, "-r", linewidth=5, label="course")
        # # plt.plot(traj_x, traj_y, marker = "o", label="course")
        # plt.plot(traj_ego_x, traj_ego_y, "-b", linewidth=2, label="trajectory")
        # plt.plot(target_point[0], target_point[1], "og", ms=5, label="target point")
        # plotVehicle(ego.x, ego.y, ego.yaw, desired_curva * dl)
        # plt.xlabel("x[m]")
        # plt.ylabel("y[m]")
        # plt.axis("equal")
        # plt.legend()
        # plt.grid(True)
        # plt.pause(0.01)

    processed_perturbation = []
    for i in range(attack_length):
        select_x = traj_ego_x[int(i*len(traj_ego_x)/attack_length)]
        select_y = traj_ego_y[int(i*len(traj_ego_x)/attack_length)]
        perturbation_x = select_x - data["objects"][obj]["observe_trace"][i][0]
        perturbation_y = select_y - data["objects"][obj]["observe_trace"][i][1]
        processed_perturbation.append([perturbation_x,perturbation_y])
    d_p = processed_perturbation-perturbation_array
    return torch.tensor(d_p).cuda() + perturbation