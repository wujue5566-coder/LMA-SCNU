import torch.nn as nn
import torch
from MFIRSTD.models.MAFEM_block.submodules_sc import DeformableAttnBlock, warp
from torch.nn.init import constant_

# ---------------------------------------------------------------------------
# MAFEM ablation modes
# ---------------------------------------------------------------------------
# MAFEM (the ``SC`` module below) bundles two functional parts:
#   * CME (cross-frame motion estimation): ``motion_branch`` + ``motion_out``
#     predict the forward/backward optical flow between adjacent frames.
#   * EnE (enhancement): ``DeformableAttnBlock`` aligns the frames with the
#     predicted flow through flow-guided deformable attention.
#
# ``ablation`` selects which parts are kept:
#   'full'   -> CME + EnE, i.e. MAFEM as used by the paper model
#   'no_cme' -> EnE only: no flow is estimated, the deformable attention falls
#               back to purely learned sampling offsets
#   'no_ene' -> CME only: frames are motion-compensated towards the centre frame
#               and directly fused, no deformable attention is used
#
# The ``collect_eng``/``bn_last``/``sigmoid`` head is shared by every variant:
# it is the MAFEM output stage and has to stay, otherwise the module could not
# produce the spatial weight consumed by the transformer.
ABLATION_FULL = 'full'
ABLATION_NO_CME = 'no_cme'
ABLATION_NO_ENE = 'no_ene'
ABLATION_MODES = (ABLATION_FULL, ABLATION_NO_CME, ABLATION_NO_ENE)


def make_model(args):
    return SC(in_channels=args.n_colors,
                        n_sequence=args.n_sequence,
                        out_channels=args.n_colors,
                        n_resblock=args.n_resblock,
                        n_feat=args.n_feat)


class SC(nn.Module):
    def __init__(self, in_channels=192, n_sequence=3, out_channels=2, ablation=ABLATION_FULL):
        super(SC, self).__init__()
        if ablation not in ABLATION_MODES:
            raise ValueError('unknown MAFEM ablation %r, expected one of %s' % (ablation, ABLATION_MODES))
        self.ablation = ablation
        self.use_cme = ablation in (ABLATION_FULL, ABLATION_NO_ENE)
        self.use_ene = ablation in (ABLATION_FULL, ABLATION_NO_CME)
        self.n_sequence = n_sequence

        self.MMA = None
        if self.use_ene:
            self.MMA = DeformableAttnBlock(n_heads=4,d_model=in_channels,n_levels=5,n_points=12,
                                           n_sequence=n_sequence, use_flow=self.use_cme)
        # self.pos_em  = PositionalEncodingPermute3D(3)
        self.motion_branch = None
        self.motion_out = None
        if self.use_cme:
            self.motion_branch = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=2*in_channels, out_channels=96//2, kernel_size=3, stride=1, padding=8, dilation=8),
                        nn.LeakyReLU(0.1,inplace=True),
                        torch.nn.Conv2d(in_channels=96//2, out_channels=64//2, kernel_size=3, stride=1, padding=16, dilation=16),
                        nn.LeakyReLU(0.1,inplace=True),
                        torch.nn.Conv2d(in_channels=64//2, out_channels=32//2, kernel_size=3, stride=1, padding=1, dilation=1),
                        nn.LeakyReLU(0.1,inplace=True),
            )
            self.motion_out = torch.nn.Conv2d(in_channels=32//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
            constant_(self.motion_out.weight.data, 0.)
            constant_(self.motion_out.bias.data, 0.)
        pad = int((n_sequence-1)/2)
        self.collect_eng = torch.nn.Conv3d(in_channels=in_channels,out_channels=384, kernel_size=(1,1,n_sequence), padding=(0,0,pad), dilation=1)
        self.bn_last = torch.nn.BatchNorm3d(num_features=384)
        self.sigmoid = torch.nn.Sigmoid()

    def compute_flow(self, frames):
        n, t, c, h, w = frames.size()
        frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_forward = self.estimate_flow(frames_1, frames_2).view(n, t-1, 2, h, w)
        flows_backward = self.estimate_flow(frames_2,frames_1).view(n, t-1, 2, h, w)

        return flows_forward,flows_backward
    def estimate_flow(self,frames_1, frames_2):
        return self.motion_out(self.motion_branch(torch.cat([frames_1, frames_2],1)))

    def align_to_center(self, frames, flow_forward, flow_backward):
        """CME-only branch: motion-compensate every frame onto the centre frame.

        The accumulated forward/backward flows are built exactly like in
        ``DeformableAttnBlock`` so that the CME-only variant sees the same
        motion compensation, only without the deformable attention.
        """
        b, t, c, h, w = frames.shape
        if t != 5:
            raise ValueError('the CME-only MAFEM ablation currently expects 5 frames, got %d' % t)
        flow_forward01 = flow_forward[:, 0]
        flow_forward12 = flow_forward[:, 1]
        flow_backward32 = flow_backward[:, 2]
        flow_backward43 = flow_backward[:, 3]
        flow_forward02 = flow_forward01 + warp(flow_forward12, flow_forward01)
        flow_backward42 = flow_backward43 + warp(flow_backward32, flow_backward43)
        return torch.stack([
            warp(frames[:, 0], flow_forward02),
            warp(frames[:, 1], flow_forward12),
            frames[:, 2],
            warp(frames[:, 3], flow_backward32),
            warp(frames[:, 4], flow_backward42),
        ], dim=1)

    def forward(self, features):
        if self.use_cme:
            flow_forward, flow_backward = self.compute_flow(features)
        else:
            # No motion estimation branch is trained in this variant.
            flow_forward, flow_backward = None, None

        if self.use_ene:
            frame, srcframe = self.MMA(features, features, flow_forward, flow_backward)
        else:
            frame = self.align_to_center(features, flow_forward, flow_backward)

        frame = self.collect_eng(frame.permute(0,2,1,3,4))
        frame = self.bn_last(frame).permute(0,2,1,3,4)
        frame = self.sigmoid(frame)

        return frame, flow_forward,flow_backward
