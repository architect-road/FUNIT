"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn

from networks import FewShotGen, GPPatchMcResDis


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class FUNITModel(nn.Module):
    def __init__(self, hp):
        super(FUNITModel, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])
        self.gen_test = copy.deepcopy(self.gen)

    def forward(self, co_data, cl_data, hp, mode):
        xa = co_data[0].cuda() # content images
        la = co_data[1].cuda() # content image lable
        xb = cl_data[0].cuda() # class images
        lb = cl_data[1].cuda() # class image lable
        if mode == 'gen_update':
            c_xa = self.gen.enc_content(xa)
            c_xb = self.gen.enc_content(xb)
            s_xa = self.gen.enc_class_model(xa) #提取原内容图片的类别特征
            s_xb = self.gen.enc_class_model(xb) #提取目标图片的类别特征
            xt = self.gen.decode(c_xa, s_xb)  # translation
            xr = self.gen.decode(c_xb, s_xb)  # reconstruction
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb) # 生成图片GAN损失
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, lb) # 重建图片GAN损失
            _, xb_gan_feat = self.dis(xb, lb)
            # _, xa_gan_feat = self.dis(xa, la)
            #特征匹配损失：包括重建图像与新类别的图像的损失（一范数）
            l_c_rec = recon_criterion(xr_gan_feat.mean(3).mean(2),
                                      xb_gan_feat.mean(3).mean(2))
            l_m_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
                                      xb_gan_feat.mean(3).mean(2))
            l_x_rec = recon_criterion(xr, xb) # 重建损失就是两张图片的像素差的平均,注意此处的损失为一张图转换后与自己原图之差
            l_adv = 0.5 * (l_adv_t + l_adv_r) # 总的GAN损失
            acc = 0.5 * (gacc_t + gacc_r) # 准确率为重建和转化的准确率的综合
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp[
                'fm_w'] * (l_c_rec + l_m_rec)) # Generator的总损失
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc
        elif mode == 'dis_update':
            xb.requires_grad_() # 设置梯度为true
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb) # 计算真实图像（类别）的GAN判别损失（real）
            l_real = hp['gan_w'] * l_real_pre # 乘上权重
            l_real.backward(retain_graph=True)
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre # 该损失指判别器输出概率值对类别图像xb求导的均方
            l_reg.backward()
            with torch.no_grad(): # 生成fake图
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb)
                xt = self.gen.decode(c_xa, s_xb)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(),
                                                                  lb) # 计算生成图像的GAN损失（fake）
            l_fake = hp['gan_w'] * l_fake_p # 乘权重
            l_fake.backward()
            l_total = l_fake + l_real + l_reg # 得到最终的判别器损失
            acc = 0.5 * (acc_f + acc_r) #判别器正确率
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
        else:
            assert 0, 'Not support operation'

    def test(self, co_data, cl_data):
        self.eval() # 为什么这里就不用with torch.no_grad了？？
        self.gen.eval()
        self.gen_test.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen.enc_content(xa)
        c_xb_current = self.gen.enc_content(xb)
        # s_xa_current = self.gen.enc_class_model(xa)
        s_xb_current = self.gen.enc_class_model(xb)
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        xr_current = self.gen.decode(c_xb_current, s_xb_current)
        c_xa = self.gen_test.enc_content(xa)
        c_xb = self.gen_test.enc_content(xb)
        # s_xa = self.gen_test.enc_class_model(xa)
        s_xb = self.gen_test.enc_class_model(xb)
        xt = self.gen_test.decode(c_xa, s_xb)
        xr = self.gen_test.decode(c_xb, s_xb)
        self.train()
        return xa, xr_current, xt_current, xb, xr, xt

    def translate_k_shot(self, co_data, cl_data, k):
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            c_xa_current = self.gen_test.enc_content(xa)
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1) # 对得到的类别编码进行平均
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
