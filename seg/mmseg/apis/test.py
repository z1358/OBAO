import os.path as osp
import pickle
import shutil
import tempfile
import datetime
import torch.nn.functional as F
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.runner import build_optimizer, build_runner
import math
from IPython import embed
from mmseg.ops import resize


import numpy as np
import kornia
import torch
import random
import torch.nn as nn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
import pdb


def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def single_gpu_our(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None,
                    args=None,
                    sim_mat_e=None,
                    sim_mat_c=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:  
        list: The prediction results.
    """

    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            # print(name)
        else:
            param.requires_grad=False

    optim_lr = args.lr
    optimizer = torch.optim.Adam(param_list, lr=optim_lr, betas=(0.9, 0.999))#Batchsize=1 now, was 8 during cityscapes training

    class_relation_type = args.class_relation_type
    non_diag_alpha = 1.0
    class_relation_loss_type = args.class_relation_loss_type
    lambda_CRP = args.lambda_CRP
    lambda_PCE = args.lambda_PCE
    e_margin = math.log(19) * args.margin  # 0.02

    out_dir = "./obao/" + "new_" + "CRP_" + str(lambda_CRP) + "PCE_" + str(lambda_PCE) + "margin" + str(args.margin) + str(datetime.datetime.now())

    print("****************************************")
    print(out_dir)
    print(class_relation_type, class_relation_loss_type, lambda_CRP, lambda_PCE)
    lr = optimizer.param_groups[0]['lr']
    print("learing rate: {} ".format(lr))
    print("****************************************")

    for i, data in enumerate(data_loader):
        model.eval()
        ema_model.eval()
        anchor_model.eval()
        with torch.no_grad():
            if len(data['img'])==14:
                result, probs, preds, true_logits = ema_model(return_loss=False, **data)
                _, probs_, _, _ = anchor_model(return_loss=False, **data)
                mask = (probs_[4][0] > 0.69).astype(np.int64) # 0.74 was the 5% quantile for cityscapes, therefore we use 0.69 here
                result = [(mask*preds[4][0] + (1.-mask)*result[0]).astype(np.int64)]
                weight = 1.
                true_logit = true_logits[4]
            else:
                result, true_logit = ema_model(return_loss=False, **data)
                weight = 1.
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result_our(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if len(data['img'])==14:
                img_id = 4 #The default size without flip 
            else:
                img_id = 0
            loss, seg_logits, features = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0],
                                                       gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0),
                                                       true_logit=None)
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        loss_trg = torch.mean(weight * loss["decode.loss_seg"])

        if lambda_CRP > 0.0:
            num_classes = seg_logits.shape[1]
            dim = features.shape[1]
            features = features.squeeze(0).permute(1, 2, 0).reshape(-1, dim)
            seg_logits = seg_logits.squeeze(0).permute(1, 2, 0).reshape(-1, num_classes)
            ent_stu = softmax_entropy(seg_logits)
            filter_ids_1 = torch.where(ent_stu < e_margin)
            z1_low_ent = features[filter_ids_1]
            logits_low_ent = seg_logits[filter_ids_1]
            y = logits_low_ent.argmax(1)

            nonZero_prototype_z1, nonZeroRows = cal_prototype(z1_low_ent, y, num_classes)

            if lambda_PCE > 0.0:
                loss_trg += F.cross_entropy(logits_low_ent, y) * lambda_PCE

            if "c" in class_relation_type or "C" in class_relation_type:
                ########################
                nonZero_prototype_z1 = F.normalize(nonZero_prototype_z1)
                sim_mat = nonZero_prototype_z1 @ nonZero_prototype_z1.T
                orig_class_simmat = sim_mat_c[nonZeroRows][:, nonZeroRows]
            else:
                sim_mat = torch.cdist(nonZero_prototype_z1, nonZero_prototype_z1)
                orig_class_simmat = sim_mat_e[nonZeroRows][:, nonZeroRows]

            if "m" in class_relation_loss_type or "M" in class_relation_loss_type:
                mse_loss = F.mse_loss(orig_class_simmat, sim_mat, reduction='mean')
                loss_trg += mse_loss * lambda_CRP
            if "t" in class_relation_loss_type or "T" in class_relation_loss_type:
                topology_loss = topology_preserving_loss(orig_class_simmat, sim_mat)
                loss_trg += topology_loss * lambda_CRP

        # torch.mean(weight*loss["decode.loss_seg"]).backward()

        loss_trg.backward()
        optimizer.step()
        optimizer.zero_grad()
        ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999)
        if args.res > 0.0:
            for nm, m  in model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<args.res).float().cuda()
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def topology_preserving_loss( s_mean_sub, s_tilde_mean_sub):
    """
    Calculate the topology preserving loss between the original similarity matrix `s`
    and the updated similarity matrix `s_tilde` based on the provided formula.

    :param s: Original similarity matrix (NxN)
    :param s_tilde: Updated similarity matrix (NxN)
    :return: Topology preserving loss value
    """
    # Mean subtraction from the similarity matrices
    # s_mean_sub = s
    # s_tilde_mean_sub = s_tilde

    # Numerator: Covariance between the original and updated similarities
    numerator = (s_mean_sub * s_tilde_mean_sub).sum()

    # Denominator: Standard deviation of the original and updated similarities
    denominator = torch.sqrt(((s_mean_sub ** 2).sum()) * ((s_tilde_mean_sub ** 2).sum()))

    # Topology preserving loss
    loss_tpl = -numerator / denominator

    return loss_tpl

def cal_prototype(z1, y1, num_classes):
    start_i = 0
    end_i = num_classes
    dim = z1.shape[1]
    current_classes_mean_z1 = torch.zeros((end_i, dim), device=z1.device)
    for i in range(start_i, end_i):
        indices = (y1 == i)
        if any(indices):
            t_z1 = z1[indices]
            mean_z1 = torch.mean(t_z1, dim=0)
            current_classes_mean_z1[i] = mean_z1

    nonZeroRows = torch.abs(current_classes_mean_z1).sum(dim=1) > 0
    nonZero_prototype_z1 = current_classes_mean_z1[nonZeroRows]

    return nonZero_prototype_z1, nonZeroRows
def single_gpu_tent(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    out_dir = "./tent/" + "new_" + str(datetime.datetime.now())
    param_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.requires_grad and ("norm" in name or "bn" in name):
                param_list.append(param)
                print (name)
            else:
                param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, _ = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result_our(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            loss, seg_logits, features = model.forward(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0), true_logit=None)
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            loss, seg_logits, features = model(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=result, true_logit=None)
            if efficient_test:
                result = np2tmp(result)
            results.append(result)


        torch.mean(loss["decode.loss_seg"]).backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_cotta(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./cotta/" + "new_" + str(datetime.datetime.now())
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            print(name)
        else:
            param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))#Batchsize=1 now, was 8 during cityscapes training
    for i, data in enumerate(data_loader):
        model.eval()
        ema_model.eval()
        anchor_model.eval()
        with torch.no_grad():
            result, probs, preds, true_logits = ema_model(return_loss=False, **data)
            _, probs_, _, _ = anchor_model(return_loss=False, **data)
            mask = (probs_[4][0] > 0.69).astype(np.int64) # 0.74 was the 5% quantile for cityscapes, therefore we use 0.69 here
            result = [(mask*preds[4][0] + (1.-mask)*result[0]).astype(np.int64)]
            weight = 1.
            true_logit = true_logits[4]

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result_our(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)
        if isinstance(result, list):
            if len(data['img'])==14:
                img_id = 4 #The default size without flip
            else:
                img_id = 0
            loss, _, _ = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0), true_logit=None)
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        torch.mean(weight*loss["decode.loss_seg"]).backward()
        optimizer.step()
        optimizer.zero_grad()
        ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999)
        for nm, m  in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<0.01).float().cuda()
                    with torch.no_grad():
                        p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    out_dir = "./baseline/"+"new_"+str(datetime.datetime.now())
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, _ = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                # pdb.set_trace()
                model.module.show_result_our(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
