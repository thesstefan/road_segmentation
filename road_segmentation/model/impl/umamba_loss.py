import numpy as np
import torch
from torch import nn, Tensor
from typing import Callable, Any, Optional, Tuple
from torch import distributed as dist



class AllGatherGrad(torch.autograd.Function):
    # stolen from pytorch lightning
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()



class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc
    
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)

    
    
class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
    
    
    
class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
    
    
def build_umamba_loss():
    loss = DC_and_CE_loss({'batch_dice': True,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': dist.is_available() and dist.is_initialized()}, {}, weight_ce=1, weight_dice=1,
                            dice_class=MemoryEfficientSoftDiceLoss)

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss

    deep_supervision_scales = deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]), axis=0))[:-1]
    weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
    weights[-1] = 0

    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    weights = weights / weights.sum()
    # now wrap the loss
    loss = DeepSupervisionWrapper(loss, weights)
    return loss