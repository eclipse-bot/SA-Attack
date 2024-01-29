import numpy as np
import torch
import logging
from torch.autograd import Variable
from torch import autograd
import copy
import random

from .pure_pursuit_direct import pure_pursuit
from .smooth_constraint import smooth_constraint
from .attack import BaseAttacker
from .loss import attack_loss
from .constraint import hard_constraint
from prediction.dataset.generate import input_data_by_attack_step

logger = logging.getLogger(__name__)


class GradientAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, attack_duration, predictor, iter_num=100, learn_rate=0.1, learn_rate_decay=20, bound=1, physical_bounds={}, smooth=0, seed_num=10):
        super().__init__(obs_length, pred_length, attack_duration, predictor)
        self.iter_num = iter_num
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.bound = bound
        self.physical_bounds = physical_bounds
        self.seed_num = seed_num
        
        self.loss = attack_loss

    def run(self, data, obj_id, **attack_opts):
        try:
            self.predictor.model.eval()
        except:
            pass

        perturbation = {"obj_id": obj_id, "loss": self.loss, "value": {}, "ready_value": {}, "attack_opts": attack_opts}
        
        lr = self.learn_rate
        if attack_opts["type"] in ["ade", "fde"]:
            lr /= 10

        if "mode" in attack_opts:
            mode = attack_opts["mode"]
        else:
            mode = "single"
        #单目标攻击和多目标攻击
        if mode == "single":
            perturbation["value"][obj_id] = None
            perturbation["ready_value"][obj_id] = None
        elif mode == "all":
            for _obj_id in data["objects"]:
                perturbation["value"][_obj_id] = None
                perturbation["ready_value"][_obj_id] = None
        elif mode == "select":
            raise NotImplementedError()

        attack_opts["loss"] = self.loss.__name__

        best_iter = None
        best_loss = 0x7fffffff
        best_out = None
        best_perturb = None
        torch.manual_seed(1)
        random.seed(1)
        #产生10次随机扰动
        for seed in range(self.seed_num):
            loss_not_improved_iter_cnt = 0

            for _obj_id in perturbation["value"]:
                #生成随机数
                #生成同一方向的随机数
                # p_p_p = pure_pursuit(data, _obj_id, self.obs_length+self.attack_duration-1, 5,50)
                #perturbation["value"][_obj_id] = Variable(torch.from_numpy(p_p_p).cuda())
                # rand_num = torch.rand(self.obs_length + self.attack_duration - 1, 2).cuda() * 2 * self.bound - self.bound
                # for i in range(self.obs_length+self.attack_duration-1):
                #     for j in range(2):
                #         if p_p_p[i][j]*rand_num[i][j] < 0:
                #             rand_num[i][j] = rand_num[i][j]*(-1)
                # perturbation["value"][_obj_id] = Variable(rand_num)

                #生成随机方向的随机数
                perturbation["value"][_obj_id] = Variable(torch.rand(self.obs_length + self.attack_duration - 1, 2).cuda() * 3 * self.bound - 1.5 * self.bound)

            # opt_Adam = torch.optim.Adam(list(perturbation["value"].values()), lr=self.learn_rate/10 if perturbation["attack_opts"]["type"] in ["ade", "fde"] else self.learn_rate)
            #每个随机扰动优化100次
            local_best_loss = 0x7fffffff
            for i in range(self.iter_num):
                if loss_not_improved_iter_cnt > 20:
                    break
                total_loss = []
                total_out = {}

                processed_perturbation = {}
                for _obj_id in perturbation["value"]:
                    perturbation["value"][_obj_id].requires_grad = True
                    #边界约束
                    hard_pert = hard_constraint(data["objects"][_obj_id]["observe_trace"], perturbation["value"][_obj_id], self.bound, self.physical_bounds)
                    # 运动约束
                    processed_perturbation[_obj_id] = smooth_constraint(data, _obj_id, hard_pert, self.obs_length + self.attack_duration - 1, 5, 50)
                    ######
                    #processed_perturbation[_obj_id] = hard_constraint(data["objects"][_obj_id]["observe_trace"], perturbation["value"][_obj_id], self.bound, self.physical_bounds)

                for k in range(self.attack_duration):
                    # construct perturbation
                    for _obj_id in processed_perturbation:
                        perturbation["ready_value"][_obj_id] = processed_perturbation[_obj_id][k:k+self.obs_length,:]
                    # construct input_data主要构建object里面的数据
                    input_data = input_data_by_attack_step(data, self.obs_length, self.pred_length, k)

                    # call predictor
                    output_data, loss = self.predictor.run(input_data, perturbation=perturbation, backward=True)
                    #记录多帧攻击输出和损失
                    total_out[k] = output_data
                    total_loss.append(loss)

                loss = sum(total_loss)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_perturb = {_obj_id:value.cpu().clone().detach().numpy() for _obj_id, value in processed_perturbation.items()}
                    best_iter = i
                    best_out = total_out

                if loss.item() < local_best_loss:
                    local_best_loss = loss.item()
                    loss_not_improved_iter_cnt = 0
                else:
                    #记录连续多少次没改善
                    loss_not_improved_iter_cnt += 1

                # opt_Adam.zero_grad()
                # loss.backward()
                # opt_Adam.step()

                perturbation["value"][_obj_id].grad

                total_grad_sum = 0
                for _obj_id in perturbation["value"]:
                    #loss=f(perturbation)
                    grad = torch.autograd.grad(loss, perturbation["value"][_obj_id], retain_graph=False, create_graph=False, allow_unused=True)[0]
                    perturbation["value"][_obj_id] = perturbation["value"][_obj_id].detach() - lr * grad
                    total_grad_sum += float(torch.sum(torch.absolute(grad)).item())
                if total_grad_sum < 0.1:
                    break

                logger.warn("Seed {} step {} finished -- loss: {}; best loss: {};".format(seed, i, loss, best_loss))

        return {
            "output_data": best_out, 
            "perturbation": best_perturb,
            "loss": best_loss,
            "obj_id": obj_id,
            "attack_opts": attack_opts,
            "attack_length": self.attack_duration
        }
