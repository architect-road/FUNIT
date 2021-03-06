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
        self.gen_a = FewShotGen(hp['gen_a']) # human domain Generator
        self.gen_b = FewShotGen(hp['gen_b']) # anime domain Generator
        self.dis_a = GPPatchMcResDis(hp['dis_a']) # human domain Discriminator
        self.dis_b = GPPatchMcResDis(hp['dis_b']) # anime domain Discriminator

        self.gen_test_a = copy.deepcopy(self.gen_a)
        self.gen_test_b = copy.deepcopy(self.gen_b)

    def forward(self, co_data, cl_data, hp, mode):
        xa = co_data[0].cuda()
        la = co_data[1].cuda()
        xb = cl_data[0].cuda()
        lb = cl_data[1].cuda()
        if mode == 'gen_update':
            # get the content and style code for human domain and anime domain
            c_xa = self.gen_a.enc_content(xa)
            c_xb = self.gen_b.enc_content(xb)
            s_xa = self.gen_a.enc_class_model(xa)
            s_xb = self.gen_b.enc_class_model(xb)
            # reconstruction
            xr_a = self.gen_a.decode(c_xa, s_xa)
            xr_b = self.gen_b.decode(c_xb, s_xb)
            # translation
            xt_a2b = self.gen_b.decode(c_xa, s_xb)
            xt_b2a = self.gen_a.decode(c_xb, s_xa)
            # recode
            c_xt_a2b = self.gen_b.enc_content(xt_a2b)
            c_xt_b2a = self.gen_a.enc_content(xt_b2a)
            s_xt_a2b = self.gen_b.enc_class_model(xt_a2b)
            s_xt_b2a = self.gen_a.enc_class_model(xt_b2a)

            ############ caculate loss ############
            # gan loss
            xt_a2b_gan_loss, xt_a2b_gan_acc, xt_a2b_gan_feat = self.dis_b.calc_gan_loss(xt_a2b, lb)
            xt_b2a_gan_loss, xt_b2a_gan_acc, xt_b2a_gan_feat = self.dis_a.calc_gan_loss(xt_b2a, la)
            xr_a_gan_loss, xr_a_gan_acc, xr_a_gan_feat = self.dis_a.calc_gan_loss(xr_a, la)
            xr_b_gan_loss, xr_b_gan_acc, xr_b_gan_feat = self.dis_b.calc_gan_loss(xr_b, lb)
            if hp['mode'] == 'B':
                gan_loss = (xt_a2b_gan_loss + xt_b2a_gan_loss + xr_a_gan_loss + xr_b_gan_loss) * 0.5
            else:
                gan_loss = xt_a2b_gan_loss + xt_b2a_gan_loss
            # feature loss
            _, xb_gan_feat = self.dis_b(xb, lb)
            _, xa_gan_feat = self.dis_a(xa, la)
            xr_feat_loss = recon_criterion(xr_a_gan_feat.mean(3).mean(2),xa_gan_feat.mean(3).mean(2)) + \
                recon_criterion(xr_b_gan_feat.mean(3).mean(2),xb_gan_feat.mean(3).mean(2))
            xt_feat_loss = recon_criterion(xt_b2a_gan_feat.mean(3).mean(2),xa_gan_feat.mean(3).mean(2)) + \
                recon_criterion(xt_a2b_gan_feat.mean(3).mean(2),xb_gan_feat.mean(3).mean(2))
            if hp['mode'] == 'B':
                feat_loss = xt_feat_loss + xr_feat_loss
            else:
                feat_loss = xt_feat_loss
            # reconstruction loss
            xa_rec_loss = recon_criterion(xr_a, xa)
            xb_rec_loss = recon_criterion(xr_b, xb)
            rec_loss = (xa_rec_loss + xb_rec_loss)
            # content loss
            content_a2b_loss = recon_criterion(c_xa,c_xt_a2b)
            content_b2a_loss = recon_criterion(c_xb,c_xt_b2a)
            content_loss = (content_a2b_loss + content_b2a_loss)
            # style loss
            style_a2b_loss = recon_criterion(s_xb, s_xt_a2b)
            style_b2a_loss = recon_criterion(s_xa, s_xt_b2a)
            style_loss = (style_a2b_loss + style_b2a_loss)
            # total loss
            total_loss = hp['gan_w'] * gan_loss + hp['r_w'] * rec_loss + hp['fm_w'] * feat_loss + hp['c_w'] * content_loss + hp['s_w'] * style_loss
            total_loss.backward()
            acc = 0.5 * (xt_a2b_gan_acc + xt_b2a_gan_acc) # the accuracy of fake image recognition
            return total_loss, gan_loss, feat_loss, rec_loss, content_loss, style_loss, acc
        elif mode == 'dis_update':
            xb.requires_grad_()
            xa.requires_grad_()
            ################# dis_a #################
            dis_a_real_loss, dis_a_real_acc, dis_a_real_resp = self.dis_a.calc_dis_real_loss(xa, la) # real loss
            dis_a_real_loss = hp['gan_w'] * dis_a_real_loss
            dis_a_real_loss.backward(retain_graph=True)
            dis_a_reg_loss = 10 * self.dis_a.calc_grad2(dis_a_real_resp, xa) # reg loss
            dis_a_reg_loss.backward()
            # fake loss
            with torch.no_grad():
                c_xb = self.gen_b.enc_content(xb)
                c_xa = self.gen_a.enc_content(xa)
                s_xa = self.gen_a.enc_class_model(xa)
                xr_a = self.gen_a.decode(c_xa, s_xa)
                xt_b2a = self.gen_a.decode(c_xb, s_xa)
            dis_at_fake_loss, dis_at_fake_acc, dis_at_fake_resp = self.dis_a.calc_dis_fake_loss(xt_b2a.detach(), la)
            dis_ar_fake_loss, dis_ar_fake_acc, dis_ar_fake_resp = self.dis_a.calc_dis_fake_loss(xr_a.detach(), la)
            if hp['mode'] == 'B':
                dis_a_fake_loss = hp['gan_w'] * (dis_at_fake_loss + dis_ar_fake_loss) * 0.5
            else:
                dis_a_fake_loss = hp['gan_w'] * dis_at_fake_loss
            dis_a_fake_loss.backward()
            ################# dis_b #################
            dis_b_real_loss, dis_b_real_acc, dis_b_real_resp = self.dis_b.calc_dis_real_loss(xb, lb) # real loss
            dis_b_real_loss.backward(retain_graph=True)
            dis_b_reg_loss = 10 * self.dis_b.calc_grad2(dis_b_real_resp, xb) #reg loss
            dis_b_reg_loss.backward()
            # fake loss
            with torch.no_grad():
                c_xa = self.gen_a.enc_content(xa)
                c_xb = self.gen_b.enc_content(xb)
                s_xb = self.gen_b.enc_class_model(xb)
                xr_b = self.gen_b.decode(c_xb, s_xb)
                xt_a2b = self.gen_b.decode(c_xa, s_xb)
            dis_bt_fake_loss, dis_bt_fake_acc, dis_bt_fake_resp = self.dis_b.calc_dis_fake_loss(xt_a2b.detach(), lb)
            dis_br_fake_loss, dis_br_fake_acc, dis_br_fake_resp = self.dis_b.calc_dis_fake_loss(xr_b.detach(), lb)
            if hp['mode'] == 'B':
                dis_b_fake_loss = hp['gan_w'] * (dis_bt_fake_loss + dis_br_fake_loss) * 0.5
            else:
                dis_b_fake_loss = hp['gan_w'] * dis_bt_fake_loss
            dis_b_fake_loss.backward()

            real_loss = (dis_a_real_loss + dis_b_real_loss)
            fake_loss = (dis_a_fake_loss + dis_b_fake_loss)
            reg_loss = (dis_a_reg_loss + dis_b_reg_loss)
            total_loss = (dis_a_fake_loss + dis_b_fake_loss + dis_a_real_loss + dis_b_real_loss + dis_a_reg_loss + dis_b_reg_loss)
            acc = 0.25 * (dis_at_fake_acc + dis_bt_fake_acc + dis_a_real_acc + dis_b_real_acc)
            # print("Dis:[fake_loss:%.2f" % fake_loss.item(),"real_loss:%.2f" % real_loss.item(),"reg_loss:%.2f]" % reg_loss.item())
            return total_loss, fake_loss, real_loss, reg_loss, acc
        else:
            assert 0, 'Not support operation'

    def test(self, co_data, cl_data):
        self.eval()
        # self.gen_a.eval()
        # self.gen_b.eval()
        # self.gen_test_a.eval()
        # self.gen_test_b.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()

        c_xa = self.gen_test_a.enc_content(xa)
        c_xb = self.gen_test_b.enc_content(xb)
        s_xa = self.gen_test_a.enc_class_model(xa)
        s_xb = self.gen_test_b.enc_class_model(xb)
        xr_a = self.gen_test_a.decode(c_xa, s_xa)
        xr_b = self.gen_test_b.decode(c_xb, s_xb)
        xt_a2b = self.gen_test_b.decode(c_xa, s_xb)
        xt_b2a = self.gen_test_a.decode(c_xb, s_xa)

        self.train()
        return xa, xr_a, xt_b2a, xb, xr_b, xt_a2b

    def translate_k_shot(self, co_data, cl_data, k):
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa = self.gen_test_a.enc_content(xa)
        if k == 1:
            s_xb = self.gen_test_b.enc_class_model(xb)
            xt_a2b = self.gen_test_b.decode(c_xa, s_xb)
        else:
            s_xb = self.gen_test_b.enc_class_model(xb)
            s_xb = s_xb.squeeze(-1).permute(1,2,0)
            s_xb = torch.nn.functional.avg_pool1d(s_xb, k)
            s_xb = s_xb.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa, s_xb)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test_b.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb = class_code.cuda()
        c_xa = self.gen_test_a.enc_content(xa)
        xt = self.gen_test_b.decode(c_xa, s_xb)
        return xt