import torch
import torch.nn as nn
import torch.nn.functional as F
from MLPDecoder import DecoderHead
from sam2.build_sam import build_sam2
import math
from einops import rearrange


# ══════════════════════════════════════════════════════════════════
# Expert 0: 空间定位专家
# ══════════════════════════════════════════════════════════════════

class E0_SpatialLocalization(nn.Module):
    """
    专注"在哪里"：文本引导的三阶段边界精炼。

    Stage1: 视觉Q → 文本K/V 交叉注意力
            软掩码对不确定区域给予更多文本引导
    Stage2: 2D深度可分离卷积（5×5），空间局部连续性修复
    Stage3: cosine对齐分数引导双路：
            高对齐 → known路径（保真）
            低对齐 → masked路径（文本引导推断）

    Fix #3: known_conv 改用 xavier_uniform 初始化，消除 eye 初始化
            导致的 pass-through 捷径问题，防止30轮后特征稳定
            时捷径失效产生的性能断崖。
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.mask_estimator = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, 1, 1),
            nn.Sigmoid(),
        )

        self.stage1_norm = nn.GroupNorm(max(1, d_model // 16), d_model)
        self.stage1_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.stage1_drop = nn.Dropout(dropout)

        self.stage2_norm = nn.GroupNorm(max(1, d_model // 16), d_model)
        self.stage2_dw = nn.Conv2d(d_model, d_model, kernel_size=5,
                                   padding=2, groups=d_model)
        self.stage2_pw = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.stage2_drop = nn.Dropout2d(dropout)

        self.stage3_norm = nn.GroupNorm(max(1, d_model // 16), d_model)
        self.known_conv = nn.Conv2d(d_model, d_model, 1)
        self.masked_conv = nn.Conv2d(d_model, d_model, 1)
        self.stage3_merge = nn.Conv2d(d_model * 2, d_model, 1)
        self.stage3_drop = nn.Dropout2d(dropout)

        self.final_norm = nn.GroupNorm(max(1, d_model // 16), d_model)
        self.final_ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, 1), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_model * 2, d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # Fix #3: 两个分支均用 xavier_uniform，放弃 eye 初始化的捷径。
        nn.init.xavier_uniform_(self.known_conv.weight)
        nn.init.zeros_(self.known_conv.bias)
        nn.init.xavier_uniform_(self.masked_conv.weight)
        nn.init.zeros_(self.masked_conv.bias)
        nn.init.xavier_uniform_(self.stage3_merge.weight)
        nn.init.zeros_(self.stage3_merge.bias)

    def forward(self, v: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        B, D, H, W = v.shape

        mask_prob = self.mask_estimator(v)

        # Stage1
        v_s1 = self.stage1_norm(v)
        v_guided = v_s1 * (0.5 + 0.5 * mask_prob)
        q = v_guided.permute(0, 2, 3, 1).reshape(B, H * W, D)
        T_kv = T.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.stage1_attn(q, T_kv, T_kv)
        attn_out = attn_out.reshape(B, H, W, D).permute(0, 3, 1, 2)
        v = v + self.stage1_drop(attn_out)

        # Stage2
        v_s2 = self.stage2_norm(v)
        v_local = self.stage2_pw(F.gelu(self.stage2_dw(v_s2)))
        v = v + self.stage2_drop(v_local)

        # Stage3
        v_s3 = self.stage3_norm(v)
        v_flat = F.normalize(
            v_s3.permute(0, 2, 3, 1).reshape(B, H * W, D), dim=-1)
        T_norm = F.normalize(T, dim=-1)
        align = torch.einsum('bnd,cd->bnc', v_flat, T_norm).max(-1)[0]
        align = align.reshape(B, 1, H, W).sigmoid()

        x_known = self.known_conv(v_s3)
        x_masked = self.masked_conv(v_s3)
        v = v + self.stage3_drop(
            self.stage3_merge(torch.cat([
                align * x_known,
                (1 - align) * x_masked,
            ], dim=1))
        )

        return v + self.final_ffn(self.final_norm(v))


# ══════════════════════════════════════════════════════════════════
# Expert 1: 噪声分离专家
# ══════════════════════════════════════════════════════════════════

class E1_NoiseDisentangle(nn.Module):
    """
    专注"去除什么"：文本引导的模态噪声分离。

    双归一化（GroupNorm + InstanceNorm2d）消除多模态批次分布差异。
    λ 由 [视觉均值, 视觉std, 文本全局] 联合预测，逐通道控制去噪强度。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * 2

        self.group_norm = nn.GroupNorm(max(1, d_model // 16), d_model)
        self.inst_norm = nn.InstanceNorm2d(d_model, affine=True)

        self.signal_branch = nn.Sequential(
            nn.Conv2d(d_model, d_ff, 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(d_ff, d_ff, 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(d_ff, d_model, 1),
        )
        self.signal_norm = nn.GroupNorm(max(1, d_model // 16), d_model)

        self.noise_branch = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 1),
            nn.Tanh(),
            nn.Conv2d(d_model // 2, d_model, 1),
        )

        self.lambda_fc = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid(),
        )

        self.output_gate = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout2d(dropout)

    def forward(self, v: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        B, D, H, W = v.shape
        residual = v
        t_global = T.mean(0).unsqueeze(0).expand(B, -1)

        v_gn = self.group_norm(v)
        v_in = self.inst_norm(v)
        v_norm = (v_gn + v_in) * 0.5

        x_signal = self.signal_norm(self.signal_branch(v_norm))
        x_noise = self.noise_branch(v_norm)

        mean = v_norm.mean(dim=[2, 3])
        std = v_norm.std(dim=[2, 3]).clamp(min=1e-6)
        lam = self.lambda_fc(
            torch.cat([mean, std, t_global], dim=-1)
        ).view(B, D, 1, 1)

        x_denoised = x_signal - lam * x_noise
        x_out = x_denoised * self.output_gate(x_denoised)
        return residual + self.drop(x_out)


# ══════════════════════════════════════════════════════════════════
# Expert 2: 多尺度频率专家
# ══════════════════════════════════════════════════════════════════

class E2_MultiScaleFreq(nn.Module):
    """
    专注"保留什么频率"：文本引导的2D多尺度频率增强。

    3×3 → 高频细节  5×5 → 中频结构  7×7 → 低频语义

    Fix #4: threshold 初始值从 0.05 提升至 0.1，alpha 初始值从
            1.0 降低至 0.5，避免训练早期特征值较小时大面积清零。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * 2
        gn = lambda c: nn.GroupNorm(max(1, c // 16), c)

        self.norm = gn(d_model)

        def _dw_block(k):
            return nn.Sequential(
                nn.Conv2d(d_model, d_model, k, padding=k // 2,
                          groups=d_model, bias=False),
                nn.Conv2d(d_model, d_model, 1, bias=False),
                gn(d_model),
                nn.GELU(),
            )

        self.scale_3 = _dw_block(3)
        self.scale_5 = _dw_block(5)
        self.scale_7 = _dw_block(7)

        # Fix #4: threshold 0.05→0.1，alpha 1.0→0.5
        self.threshold_3 = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.1)
        self.threshold_5 = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.1)
        self.threshold_7 = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.1)
        self.alpha_3 = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.5)
        self.alpha_5 = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.5)
        self.alpha_7 = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.5)

        self.low2high_gate = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False), nn.Sigmoid())
        self.high2low_proj = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.sharp_alpha = nn.Parameter(torch.tensor(0.1))
        self.mid_refine = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False), nn.Sigmoid())

        self.film_3 = nn.Linear(d_model, d_model * 2)
        self.film_5 = nn.Linear(d_model, d_model * 2)
        self.film_7 = nn.Linear(d_model, d_model * 2)
        for film in [self.film_3, self.film_5, self.film_7]:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            film.bias.data[:d_model] = 1.0

        self.spatial_gate_local = nn.Sequential(
            nn.Conv2d(d_model * 3, d_model // 2, 3, padding=1, bias=False),
            gn(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, 3, 1, bias=False),
        )
        self.spatial_gate_global = nn.Linear(d_model, 3)
        nn.init.zeros_(self.spatial_gate_global.weight)
        nn.init.ones_(self.spatial_gate_global.bias)

        self.ch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

        self.ffn_norm = gn(d_model)
        self.ffn_gate = nn.Conv2d(d_model, d_ff, 1, bias=False)
        self.ffn_val = nn.Conv2d(d_model, d_ff, 1, bias=False)
        self.ffn_proj = nn.Conv2d(d_ff, d_model, 1, bias=False)
        self.drop = nn.Dropout2d(dropout)

    @staticmethod
    def _soft_threshold(x, thr, alpha):
        return torch.sign(x) * alpha * F.relu(torch.abs(x) - torch.abs(thr))

    @staticmethod
    def _film(x, params):
        D = x.size(1)
        scale = params[:D].view(1, D, 1, 1)
        shift = params[D:].view(1, D, 1, 1)
        return x * scale + shift

    def forward(self, v: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        B, D, H, W = v.shape
        residual = v
        t_global = T.mean(0)
        v_norm = self.norm(v)

        s3 = self.scale_3(v_norm)
        s5 = self.scale_5(v_norm)
        s7 = self.scale_7(v_norm)

        s3 = self._film(s3, self.film_3(t_global))
        s5 = self._film(s5, self.film_5(t_global))
        s7 = self._film(s7, self.film_7(t_global))

        s3 = self._soft_threshold(s3, self.threshold_3, self.alpha_3)
        s5 = self._soft_threshold(s5, self.threshold_5, self.alpha_5)
        s7 = self._soft_threshold(s7, self.threshold_7, self.alpha_7)

        s3 = s3 * self.low2high_gate(s7)
        s7 = s7 + self.sharp_alpha * self.high2low_proj(s3)
        s5 = s5 * self.mid_refine(s3 + s7)

        stack = torch.cat([s3, s5, s7], dim=1)
        w_local = self.spatial_gate_local(stack)
        w_global = self.spatial_gate_global(t_global).view(1, 3, 1, 1)
        w_map = F.softmax(w_local + w_global, dim=1)
        v_ms = (w_map[:, 0:1] * s3 +
                w_map[:, 1:2] * s5 +
                w_map[:, 2:3] * s7)

        v_ms = v_ms * self.ch_attn(v_ms).view(B, D, 1, 1)
        x = residual + self.drop(v_ms)

        x_ffn = self.ffn_norm(x)
        x = x + self.drop(
            self.ffn_proj(F.silu(self.ffn_gate(x_ffn)) * self.ffn_val(x_ffn))
        )
        return x


# ══════════════════════════════════════════════════════════════════
# 数值防御对齐损失项定义
# 💡 数值防御重构损失：平滑高斯散度对齐 + 软隔离 Hinge 损失（彻底根除断崖下跌）
# ══════════════════════════════════════════════════════════════════

def ot_hinge_joint_loss(feat_v: torch.Tensor, feat_t: torch.Tensor, margin: float = 0.15):
    """
    通过引入两阶段数值防御保护，规避训练后期的梯度断崖风险：
    1. ot_intra: 使用 Huber-style 平滑控制二阶方差差异，限制过大拉力，防止强行扭曲视觉特征。
    2. loss_hinge: 使用 Soft-margin Softplus 代替传统的刚性平停 ReLU，确保后期对齐后依然保持微弱梯度流，防止路由器骤停。
    """
    # ── 1. 宏观对齐路：平滑鲁棒型对角高斯差异测量 ──
    mu_v, var_v = feat_v.mean(0), feat_v.var(0, unbiased=False)
    mu_t, var_t = feat_t.mean(0), feat_t.var(0, unbiased=False)

    mean_diff = (mu_v - mu_t) ** 2
    # 采用平滑 Huber 机制保护方差开根号求导在后期的不稳定性
    std_v, std_t = torch.sqrt(var_v + 1e-5), torch.sqrt(var_t + 1e-5)
    std_diff = F.huber_loss(std_v, std_t, reduction='sum', delta=0.5)

    loss_ot_intra = mean_diff.sum() + std_diff

    # ── 2. 微观对齐路：全时平滑软隔离损失 (Soft-margin Hinge via Softplus) ──
    sim_vc = feat_v @ feat_t.T
    sim_max = sim_vc.max(dim=-1)[0]
    sim_mean = sim_vc.mean(dim=-1)

    # 核心修正：用 softplus(x) 替换 relu(x)。当距离足够大时，损失极小但不为 0，依然提供微弱的边界维持梯度
    loss_hinge = F.softplus(margin - (sim_max - sim_mean)).mean()

    return loss_ot_intra, loss_hinge


# ══════════════════════════════════════════════════════════════════
# 辅助投影组件：SpatialVisualProjector
# ══════════════════════════════════════════════════════════════════

class SpatialVisualProjector(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.GroupNorm(max(1, embed_dim // 16), embed_dim),
            nn.ReLU(inplace=True),
        )
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor):
        feat_spatial = self.spatial_proj(x)
        feat_global = F.normalize(self.global_head(feat_spatial), dim=-1)
        return feat_spatial, feat_global


# ══════════════════════════════════════════════════════════════════
# HardNegAwareGate 门控路由
# Fix #1: 统一训练/推理输入维度，消除行为不一致问题。
# ══════════════════════════════════════════════════════════════════

class HardNegAwareGate(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Fix #1: 统一为 embed_dim*2+1，消除 train/eval 维度分歧
        self.gate = nn.Linear(embed_dim * 2 + 1, num_experts, bias=True)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def forward(self,
                v_g: torch.Tensor,  # [B, D]
                feat_t: torch.Tensor,  # [C, D]
                align: torch.Tensor,  # [B, 1]
                ) -> torch.Tensor:
        B = v_g.size(0)
        t_g = feat_t.mean(0, keepdim=True).expand(B, -1)

        logits = self.gate(torch.cat([v_g, t_g, align], dim=-1))

        # 训练时加极轻噪声（不改变维度），防止门控过早锁定
        if self.training:
            logits = logits + 0.01 * torch.randn_like(logits)

        topk_v, topk_i = logits.topk(self.top_k, dim=-1)
        w = torch.zeros_like(logits)
        w.scatter_(1, topk_i, F.softmax(topk_v, dim=-1))
        return w


# ══════════════════════════════════════════════════════════════════
# ✅ 升级：文本特征空间精细重投影对比头（对应原 PMRL contra_head_t — Bug2 修复）
# ══════════════════════════════════════════════════════════════════

class TextClassEmbedder(nn.Module):
    """
    接收经过自注意力/MoE强化后的文本表征，对其进行二次规范化与通道深层投影。
    包含 LayerNorm 稳定大盘分布，以及线性对比头（Contra Head）提取强辨别特征并进行统一的 L2 约束。
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-12)
        self.contra_head = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, T_enhanced: torch.Tensor) -> torch.Tensor:
        """
        Args: T_enhanced : [C, embed_dim]
        Returns: [C, embed_dim] 经过规范化重构与 L2 归一化后的文本表征
        """
        x = self.ln(T_enhanced)
        x = self.contra_head(x)
        return F.normalize(x, dim=-1)


# ══════════════════════════════════════════════════════════════════
# 文本-视觉联合对齐安全主控架构
# Fix #2: hinge_loss_w 0.2→0.05，避免4尺度累积后过度优化 margin 约束的问题。
# ══════════════════════════════════════════════════════════════════

class BrainTumorTextVisualMoE(nn.Module):
    """
    脑肿瘤分割专用文本-视觉 MoE 融合模块（完全重构安全防御版）。
    """

    def __init__(self,
                 in_channels,
                 embed_dim: int = 256,
                 num_classes: int = 3,
                 top_k: int = 2,
                 eta=None,
                 load_loss_w: float = 0.01,
                 ot_loss_w: float = 0.005,  # 独立控制宏观 Wasserstein，配合主干时缩减系数
                 hinge_loss_w: float = 0.1,  # 独立控制微观 Hinge
                 margin: float = 0.15):
        super().__init__()
        self.top_k = top_k
        self.eta = eta or [0.1, 0.2, 0.3, 0.4]
        self.load_loss_w = load_loss_w
        self.ot_loss_w = ot_loss_w
        self.hinge_loss_w = hinge_loss_w
        self.margin = margin
        self.num_experts = 3

        self.vis_projs = nn.ModuleList([
            SpatialVisualProjector(c, embed_dim) for c in in_channels
        ])

        self.experts = nn.ModuleList([
            E0_SpatialLocalization(embed_dim),  # Fix #3 已在类内完成
            E1_NoiseDisentangle(embed_dim),
            E2_MultiScaleFreq(embed_dim),  # Fix #4 已在类内完成
        ])

        self.heads = nn.ModuleList([
            nn.Conv2d(embed_dim, num_classes, 1) for _ in range(self.num_experts)
        ])

        self.gate = HardNegAwareGate(embed_dim, self.num_experts, top_k)

    def _route(self,
               feat_spatial: torch.Tensor,
               feat_global: torch.Tensor,
               feat_t: torch.Tensor,
               ) -> torch.Tensor:
        B = feat_spatial.size(0)
        v_flat = F.normalize(
            feat_spatial.permute(0, 2, 3, 1).reshape(B, -1, feat_spatial.size(1)), dim=-1)
        T_norm = F.normalize(feat_t, dim=-1)
        align = (torch.einsum('bnd,cd->bnc', v_flat, T_norm)
                 .max(-1)[0].mean(-1, keepdim=True))  # [B, 1]
        return self.gate(feat_global, feat_t, align)

    def forward(self, fuse_list, T, decoder_out, epoch=0, total_epochs=100):
        B, _, H_out, W_out = decoder_out.shape
        out = decoder_out
        target = 1.0 / self.num_experts

        load_loss = decoder_out.new_tensor(0.0)
        loss_ot = decoder_out.new_tensor(0.0)
        loss_hinge = decoder_out.new_tensor(0.0)

        # 随时间动态余弦衰减对齐损失权重，后期将主导权彻底交回 Dice 损失
        time_decay = math.cos(epoch / max(total_epochs, 1) * math.pi / 2)
        current_ot_w = self.ot_loss_w * time_decay
        current_hinge_w = self.hinge_loss_w * time_decay

        for i, (fuse, vis_proj) in enumerate(zip(fuse_list, self.vis_projs)):
            feat_spatial, feat_global = vis_proj(fuse)

            if self.training:
                ot_intra, hinge = ot_hinge_joint_loss(feat_global, T, self.margin)
                loss_ot = loss_ot + ot_intra
                loss_hinge = loss_hinge + hinge

            w = self._route(feat_spatial, feat_global, T)

            if self.training:
                load_loss = load_loss + ((w.mean(0) - target) ** 2).sum()

            active_cols = (w > 0).any(dim=0)
            e_outs = [None] * self.num_experts
            for e in range(self.num_experts):
                if active_cols[e]:
                    e_outs[e] = self.experts[e](feat_spatial, T)

            s = torch.zeros(B, self.heads[0].out_channels, H_out, W_out,
                            device=feat_spatial.device, dtype=feat_spatial.dtype)
            for e in range(self.num_experts):
                if e_outs[e] is not None:
                    head_out = F.interpolate(
                        self.heads[e](e_outs[e]),
                        size=(H_out, W_out),
                        mode='bilinear', align_corners=False,
                    )
                    s = s + w[:, e].view(B, 1, 1, 1) * head_out

            out = out + self.eta[i] * s

        if self.training:
            return out, load_loss * self.load_loss_w, loss_ot * current_ot_w, loss_hinge * current_hinge_w
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.GroupNorm(max(1, out_planes // 16), out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super().__init__()
        self.block = blk
        for attr in ['attn', 'mlp']:
            if hasattr(blk, attr):
                if hasattr(getattr(blk, attr), 'qkv'):
                    dim = getattr(blk, attr).qkv.in_features
                    break
        else:
            dim = blk.norm1.weight.shape[0]
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 24), nn.ReLU(), nn.Linear(24, dim),
        )

    def forward(self, x):
        return self.block(x + self.prompt_learn(x))


class GatedScanUnit(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.in_proj  = nn.Linear(dim, dim * 2, bias=False)
        self.conv1d   = nn.Conv1d(dim, dim, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=dim, bias=True)
        self.x_proj   = nn.Linear(dim, d_state * 2 + dim, bias=False)
        self.dt_proj  = nn.Linear(dim, dim, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(dim, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_main.transpose(1, 2))[..., :L].transpose(1, 2)
        x_act  = F.silu(x_conv)
        B_ssm, C_ssm, dt_raw = torch.split(
            self.x_proj(x_act), [self.d_state, self.d_state, D], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        A  = -torch.exp(self.A_log.float())
        dBx = B_ssm.unsqueeze(2) * x_act.unsqueeze(-1)
        y   = (dBx * C_ssm.unsqueeze(2)).sum(-1)
        y   = y + x_act * self.D
        y   = y * F.silu(z)
        return self.out_proj(y)


TARGET_SIZE = 32


class CrossScanBlock(nn.Module):
    """
    [Bug-3] 正向 ssm_fwd / 反向 ssm_rev 独立参数，分别学习两模态序列方向的转移规律。
    [Imp-6] 超过 TARGET_SIZE 时改用网格均匀采样（grid_sample），保留肿瘤边界细节。
    [Imp-7] 融合输出增加残差投影，稳定浅层梯度。
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.lin1  = nn.Conv2d(dim, dim, 1, bias=False)
        self.lin2  = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=True)
        # [Bug-3] 独立正 / 反向 SSM
        self.ssm_fwd = GatedScanUnit(dim, d_state)
        self.ssm_rev = GatedScanUnit(dim, d_state)
        # [Imp-7] 残差输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(max(1, dim // 16), dim),
        )

    @staticmethod
    def _grid_sample_tokens(feat: torch.Tensor, target: int) -> torch.Tensor:
        """均匀网格采样到 target×target，保留空间分布，对肿瘤边界更友好。"""
        grid_y = torch.linspace(-1, 1, target, device=feat.device)
        grid_x = torch.linspace(-1, 1, target, device=feat.device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid   = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(feat.size(0), -1, -1, -1)
        return F.grid_sample(feat, grid, mode='bilinear',
                             align_corners=True, padding_mode='border')

    def forward(self, Fm1: torch.Tensor, Fm2: torch.Tensor) -> torch.Tensor:
        Fln1 = self.lin1(self.norm1(Fm1))
        Fln2 = self.lin2(self.norm2(Fm2))
        Fs1  = F.silu(self.conv1(Fln1))
        Fs2  = F.silu(self.conv2(Fln2))

        B, C, H, W = Fs1.shape
        need = (H > TARGET_SIZE or W > TARGET_SIZE)
        # [Imp-6] 网格均匀采样替代 avg_pool
        s1 = self._grid_sample_tokens(Fs1, TARGET_SIZE) if need else Fs1
        s2 = self._grid_sample_tokens(Fs2, TARGET_SIZE) if need else Fs2
        Hs, Ws = s1.shape[-2:]
        L = Hs * Ws

        t1 = rearrange(s1, 'b c h w -> b (h w) c')
        t2 = rearrange(s2, 'b c h w -> b (h w) c')

        # [Bug-3] 正向 / 反向使用独立 SSM
        fwd_out_12 = self.ssm_fwd(torch.cat([t1, t2], dim=1))
        fwd_out_21 = self.ssm_fwd(torch.cat([t2, t1], dim=1))
        rev_out_21 = self.ssm_rev(torch.cat([t2.flip(1), t1.flip(1)], dim=1))
        rev_out_12 = self.ssm_rev(torch.cat([t1.flip(1), t2.flip(1)], dim=1))

        fcm1 = (fwd_out_12[:, :L] + rev_out_21[:, L:].flip(1)) * 0.5
        fcm2 = (fwd_out_21[:, :L] + rev_out_12[:, L:].flip(1)) * 0.5

        fcm1 = rearrange(fcm1, 'b (h w) c -> b c h w', h=Hs, w=Ws)
        fcm2 = rearrange(fcm2, 'b (h w) c -> b c h w', h=Hs, w=Ws)

        if need:
            fcm1 = F.interpolate(fcm1, (H, W), mode='bilinear', align_corners=False)
            fcm2 = F.interpolate(fcm2, (H, W), mode='bilinear', align_corners=False)

        fused = fcm1 * F.silu(Fln2) + fcm2 * F.silu(Fln1)
        return self.out_proj(fused) + fused


# ══════════════════════════════════════════════════════════════════
# MultiGrainEdgeExpert — 多粒度边界感知专家
# 适配脑肿瘤：增强肿瘤核心(TC) / 增强肿瘤(ET) 的锐利边界
# ══════════════════════════════════════════════════════════════════

class MultiGrainEdgeExpert(nn.Module):
    """
    多尺度深度可分离卷积（3×3 / 5×5 / 7×7）捕获高/中/低频边界信息，
    SE 通道注意力自适应融合，无需全局降采样，计算量可控。
    """

    def __init__(self, dim: int):
        super().__init__()
        gn = lambda c: nn.GroupNorm(max(1, c // 16), c)

        self.align1 = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), gn(dim))
        self.align2 = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), gn(dim))

        def _dw(k):
            return nn.Sequential(
                nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False),
                nn.Conv2d(dim, dim, 1, bias=False),
                gn(dim), nn.GELU(),
            )

        self.dw3 = _dw(3)  # 高频细节：肿瘤轮廓锐利边缘
        self.dw5 = _dw(5)  # 中频结构：瘤周水肿过渡区
        self.dw7 = _dw(7)  # 低频语义：整体肿瘤形态

        self.scale_w = nn.Parameter(torch.ones(3) / 3)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(dim, dim // 4), nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim), nn.Sigmoid(),
        )
        self.out_norm = gn(dim)

    def forward(self, Fm1: torch.Tensor, Fm2: torch.Tensor) -> torch.Tensor:
        x  = F.gelu(self.align1(Fm1) + self.align2(Fm2))
        s3 = self.dw3(x)
        s5 = self.dw5(x)
        s7 = self.dw7(x)
        w  = F.softmax(self.scale_w, dim=0)
        ms = w[0] * s3 + w[1] * s5 + w[2] * s7
        B, D = ms.shape[:2]
        ms = ms * self.ca(ms).view(B, D, 1, 1)
        return self.out_norm(ms)


# ══════════════════════════════════════════════════════════════════
# SpectraFuseExpert — 谱域频率互补融合专家
# 适配脑肿瘤：T2/FLAIR 低频水肿 ↔ T1ce 高频增强边界 互补增强
# ══════════════════════════════════════════════════════════════════

class SpectraFuseExpert(nn.Module):
    """
    将两路模态特征分解为低频（全局均值）与高频（局部残差）分量，
    跨模态互补门控后重建，对 T1ce 增强区（ET）与 FLAIR 水肿区（WT）特别有效。
    """

    def __init__(self, dim: int):
        super().__init__()
        gn = lambda c: nn.GroupNorm(max(1, c // 16), c)

        self.low_proj1 = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(),
                                       nn.Linear(dim // 2, dim))
        self.low_proj2 = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(),
                                       nn.Linear(dim // 2, dim))

        self.high_dw1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False), gn(dim), nn.GELU())
        self.high_dw2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False), gn(dim), nn.GELU())

        self.gate_12 = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False), nn.Sigmoid())
        self.gate_21 = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False), nn.Sigmoid())

        self.merge = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False), gn(dim), nn.GELU())

    def forward(self, Fm1: torch.Tensor, Fm2: torch.Tensor) -> torch.Tensor:
        B, D, H, W = Fm1.shape
        low1 = self.low_proj1(Fm1.mean(dim=[2, 3])).view(B, D, 1, 1)
        low2 = self.low_proj2(Fm2.mean(dim=[2, 3])).view(B, D, 1, 1)
        hi1  = self.high_dw1(Fm1 - low1)
        hi2  = self.high_dw2(Fm2 - low2)
        gate12  = self.gate_12(torch.cat([hi1, hi2], dim=1))
        gate21  = self.gate_21(torch.cat([hi2, hi1], dim=1))
        hi1_aug = hi1 + gate12 * hi2
        hi2_aug = hi2 + gate21 * hi1
        out1 = low1 + hi1_aug
        out2 = low2 + hi2_aug
        return self.merge(torch.cat([out1, out2], dim=1))



class OmniScanMoE(nn.Module):
    """
    脑肿瘤多模态跨模态融合 MoE 模块（异质四专家版）。

    专家分工：
      0 - CrossScanBlock(d_state=16)  全局长程，WT 弥散边界
      1 - CrossScanBlock(d_state=8)   局部短程，TC 紧致核心
      2 - MultiGrainEdgeExpert        多尺度卷积，ET 锐利边缘
      3 - SpectraFuseExpert           频率互补，T1ce/FLAIR 对比增强

    核心修复：
      [Bug-1] _loss_cons 维度错误 → W[:, i].view(B,1,1,1)
      [Bug-2] _loss_div detach 截断梯度 → 移除
      [Bug-3] CrossScanBlock 正/反向 SSM 独立（已在内部修复）

    核心改进：
      [Imp-1] 门控加入 std_pool，感知多模态活跃度差异
      [Imp-2] 噪声随 epoch 退火，前期探索后期收敛
      [Imp-4] stop-grad 量纲归一化，防止单项损失主导
      [Imp-5] epoch 状态通过 set_epoch() + register_buffer 内置管理
    """

    def __init__(self, dim: int, n_experts: int = 4, top_k: int = 2,
                 d_state: int = 16):
        super().__init__()
        assert n_experts == 4, "OmniScanMoE 固定使用四位异质专家"
        self.N   = n_experts
        self.K   = top_k
        self.eps = 1e-6

        # [Imp-3] 异质四专家
        self.experts = nn.ModuleList([
            CrossScanBlock(dim, d_state=d_state),       # 全局 SSM
            CrossScanBlock(dim, d_state=d_state // 2),  # 局部 SSM
            MultiGrainEdgeExpert(dim),                  # 边界卷积
            SpectraFuseExpert(dim),                     # 频率感知
        ])

        # [Imp-1] 门控输入：avg + max + std，共 6C
        gate_in = 2 * dim * 3
        self.gate_fc  = nn.Linear(gate_in, n_experts, bias=False)
        self.noise_fc = nn.Linear(gate_in, n_experts, bias=False)
        nn.init.normal_(self.gate_fc.weight,  std=0.02)
        nn.init.normal_(self.noise_fc.weight, std=0.02)

        # 融合残差投影
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(max(1, dim // 16), dim),
        )

        # [Imp-5] epoch 内置管理
        self.register_buffer('_epoch',        torch.tensor(0,   dtype=torch.float32))
        self.register_buffer('_total_epochs', torch.tensor(100, dtype=torch.float32))

    def set_epoch(self, epoch: int, total_epochs: int = 100):
        """训练循环每 epoch 开始调一次，驱动噪声退火与损失权重调度。"""
        self._epoch.fill_(epoch)
        self._total_epochs.fill_(total_epochs)

    # ── 门控 ──────────────────────────────────────────────────────
    def _gating(self, Fm1: torch.Tensor, Fm2: torch.Tensor):
        Fmc = torch.cat([Fm1, Fm2], dim=1)
        avg = F.adaptive_avg_pool2d(Fmc, 1)
        mx  = F.adaptive_max_pool2d(Fmc, 1)
        # [Imp-1] std_pool：捕捉 MRI 多序列响应差异
        std = Fmc.flatten(2).std(dim=-1, keepdim=True).unsqueeze(-1)
        Fg  = torch.cat([avg, mx, std], dim=1).squeeze(-1).squeeze(-1)  # [B, 6C]

        logits = self.gate_fc(Fg)

        if self.training:
            # [Imp-2] 噪声退火：progress 0→1 时 noise_scale 0.5→0
            progress    = (self._epoch / self._total_epochs.clamp(min=1)).clamp(0, 1)
            noise_scale = 0.1 * (1.0 - progress.float())  # 初始噪声降低，防止早期路由混乱
            noise       = torch.randn_like(logits) * F.softplus(self.noise_fc(Fg))
            logits      = logits + noise * noise_scale

        logits   = logits - logits.max(dim=-1, keepdim=True)[0]
        W_dense  = F.softmax(logits, dim=-1) + self.eps

        topk_vals, topk_idx = logits.topk(self.K, dim=-1)
        W_sparse = torch.zeros_like(logits)
        W_sparse.scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1) + self.eps)
        return W_sparse, W_dense

    # ── 损失项 ────────────────────────────────────────────────────
    def _loss_wb(self, W: torch.Tensor) -> torch.Tensor:
        """负载均衡损失：惩罚专家利用率变异系数²"""
        W_d  = W.detach() + self.eps
        mean = W_d.mean(0) + self.eps
        std  = W_d.std(0)  + self.eps
        return ((std / mean) ** 2).mean()

    def _loss_div(self, outs: list) -> torch.Tensor:
        """多样性损失：惩罚专家输出相似度。
        采用单侧 detach：以 j 为锚点引导 i 远离，避免双向对抗梯度冲突。
        """
        if len(outs) <= 1:
            return torch.tensor(0.0, device=outs[0].device)
        flat = []
        for o in outs:
            feat = F.adaptive_avg_pool2d(o, 4).flatten(1)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + self.eps)
            flat.append(feat)
        loss, cnt = 0.0, 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                # 单侧 detach：j 作锚点，梯度只流向 i，避免双向对抗
                sim = F.cosine_similarity(flat[i], flat[j].detach(), dim=-1)
                loss += sim.clamp(-0.99, 0.99).mean()
                cnt  += 1
        return loss / max(cnt, 1)

    def _loss_cons(self, expert_outs: list,
                   W: torch.Tensor,
                   dummy_feat: torch.Tensor) -> torch.Tensor:
        """一致性损失：各专家向加权共识靠拢。[Bug-1] 维度修复。"""
        B = dummy_feat.size(0)
        processed = [
            o if o is not None else torch.zeros_like(dummy_feat) + self.eps
            for o in expert_outs
        ]
        F_cons = sum(
            W[:, i].view(B, 1, 1, 1) * processed[i]   # [Bug-1] 正确维度
            for i in range(self.N)
        )
        loss = 0.0
        for i in range(self.N):
            diff = processed[i] - F_cons
            loss += (W[:, i].view(B, 1, 1, 1) * (diff ** 2)).mean()
        return loss

    # ── 前向 ──────────────────────────────────────────────────────
    def forward(self, Fm1: torch.Tensor, Fm2: torch.Tensor):
        W_sparse, W_dense = self._gating(Fm1, Fm2)

        active_idx  = (W_sparse > 0).any(dim=0).nonzero(as_tuple=True)[0].tolist()
        expert_outs = [None] * self.N
        for i in active_idx:
            expert_outs[i] = self.experts[i](Fm1, Fm2)

        Fmf = torch.zeros_like(Fm1)
        for i in range(self.N):
            if expert_outs[i] is not None:
                Fmf = Fmf + W_sparse[:, i].view(-1, 1, 1, 1) * expert_outs[i]

        # 残差投影
        Fmf = self.fuse_proj(Fmf) + Fm1

        active_outs = [o for o in expert_outs if o is not None]
        if not active_outs:
            return Fmf, torch.tensor(0.0, device=Fm1.device)

        if self.training:
            lam = math.cos(
                self._epoch.item() / max(self._total_epochs.item(), 1) * math.pi / 2
            )
            L_wb   = self._loss_wb(W_dense)
            L_div  = self._loss_div(active_outs)
            L_cons = self._loss_cons(expert_outs, W_dense, dummy_feat=Fm1)

            # 小系数加权原始损失，避免 MCCM 主导总损失
            # L_wb 自然量级 ~0.1-0.5，直接用；L_div / L_cons 乘小系数压制
            loss_mccm = (0.1 * L_wb
                         + 0.05 * lam * L_div
                         + 0.05 * (1 - lam) * L_cons)
            if torch.isnan(loss_mccm):
                loss_mccm = torch.tensor(0.0, device=Fm1.device)
        else:
            loss_mccm = torch.tensor(0.0, device=Fm1.device)

        return Fmf, loss_mccm


class TextSpecializedMoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.skip_threshold = nn.Parameter(torch.tensor(0.6))

    def forward(self, x):
        B, N, C = x.shape
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        top_weights, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        max_score = gate_weights.max(dim=-1)[0]
        skip_mask = max_score < self.skip_threshold

        x_flat = x.reshape(-1, C)
        top_indices_flat = top_indices.reshape(-1, self.top_k)
        top_weights_flat = top_weights.reshape(-1, self.top_k)

        output_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            eid = top_indices_flat[:, k]
            w = top_weights_flat[:, k].unsqueeze(-1)
            for ei in eid.unique():
                mask = (eid == ei)
                output_flat[mask] += w[mask] * self.experts[ei](x_flat[mask])

        output = output_flat.view(B, N, C)
        output = torch.where(skip_mask.unsqueeze(-1), x, output)
        return output


class TextOnlyMoEBlock(nn.Module):
    def __init__(self, dim, num_heads, num_experts=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_text = nn.LayerNorm(dim)
        self.text_moe = TextSpecializedMoE(dim, num_experts)

    def forward(self, text_feat):
        feat = self.norm1(text_feat)
        attn_out, _ = self.attn(feat, feat, feat)
        text_feat = text_feat + self.dropout(attn_out)
        text_feat = text_feat + self.dropout(
            self.text_moe(self.norm_text(text_feat)))
        return text_feat


class TextOnlyMoEModel(nn.Module):
    def __init__(self, dim=512, num_heads=8, num_layers=6,
                 num_experts=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TextOnlyMoEBlock(dim, num_heads, num_experts, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, text_feat):
        for layer in self.layers:
            text_feat = layer(text_feat)
        return self.final_norm(text_feat)


class TextBranch(nn.Module):
    def __init__(self, conch_model, class_names: list,
                 embed_dim: int = 256, num_context_tokens: int = 16):
        super().__init__()
        self.class_names = class_names
        self.C = len(class_names)
        self.M = num_context_tokens
        self.D = embed_dim

        text_enc = conch_model.text
        self.token_embedding = text_enc.token_embedding
        self.positional_embedding = text_enc.positional_embedding
        self.resblocks = text_enc.transformer.resblocks
        self.ln_final = text_enc.ln_final

        for p in text_enc.parameters():
            p.requires_grad = False

        raw_proj = getattr(conch_model, "text_projection", None)
        if raw_proj is None:
            raw_proj = getattr(text_enc, "text_projection", None)
        if raw_proj is None:
            raise AttributeError("找不到 text_projection，请确认 CONCH 结构。")

        if isinstance(raw_proj, (nn.Parameter, torch.Tensor)):
            W_clip = raw_proj.detach()
        elif isinstance(raw_proj, nn.Linear):
            W_clip = raw_proj.weight.detach().T
        else:
            raise TypeError(f"不支持的 projection 类型: {type(raw_proj)}")

        self.register_buffer("W_clip", W_clip)
        D_tok = self.token_embedding.embedding_dim
        D_clip = self.W_clip.shape[1]

        self.W_proj = nn.Linear(D_clip, embed_dim, bias=False)
        nn.init.normal_(self.W_proj.weight, std=0.02)

        self.context_tokens = nn.Parameter(
            torch.empty(self.C, num_context_tokens, D_tok))
        nn.init.normal_(self.context_tokens, std=0.02)
        self._tokenize_class_names()

    @torch.no_grad()
    def _tokenize_class_names(self):
        try:
            from CONCH.conch.open_clip_custom import get_tokenizer, tokenize
            tokenizer = get_tokenizer()
            ids = tokenize(texts=self.class_names, tokenizer=tokenizer)
        except Exception:
            ids = torch.randint(0, 100, (self.C, 77))
        self.register_buffer("class_token_ids", ids)
        self.context_length = ids.shape[1]
        self.register_buffer("eot_positions_base", ids.argmax(dim=-1))

    def _build_sequences(self) -> torch.Tensor:
        base = self.token_embedding(self.class_token_ids)
        bos = base[:, :1, :]
        rest = base[:, 1:, :]
        x = torch.cat([bos, self.context_tokens, rest], dim=1)
        max_pos = self.positional_embedding.shape[0]
        L = min(x.shape[1], max_pos)
        x = x[:, :L, :]
        x = x + self.positional_embedding[:L].unsqueeze(0)
        return x

    @staticmethod
    def _causal_mask(seq_len, device):
        mask = torch.empty(seq_len, seq_len, device=device)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self) -> torch.Tensor:
        x = self._build_sequences()
        C, L, _ = x.shape
        attn_mask = self._causal_mask(L, x.device)
        h = x.permute(1, 0, 2)
        for block in self.resblocks:
            h = block(h, attn_mask=attn_mask)
        h = h.permute(1, 0, 2)
        h = self.ln_final(h)
        eot_idx = (self.eot_positions_base + self.M).clamp(max=L - 1)
        h_eot = h[torch.arange(C, device=h.device), eot_idx]
        h_clip = h_eot @ self.W_clip
        return self.W_proj(h_clip)


# ══════════════════════════════════════════════════════════════════
# ✅ 修正连接点：将重构后的 TextClassEmbedder 闭环串联在文本增强流末端
# ══════════════════════════════════════════════════════════════════

class TextBranchWithMoE(nn.Module):
    """TextBranch + TextOnlyMoEModel + TextClassEmbedder (Bug2 修复)"""

    def __init__(self, conch_model, class_names,
                 embed_dim=256, num_context_tokens=16,
                 moe_layers=2, num_heads=8, num_experts=8):
        super().__init__()
        self.text_branch = TextBranch(
            conch_model, class_names, embed_dim, num_context_tokens)
        self.moe_encoder = TextOnlyMoEModel(
            dim=embed_dim, num_heads=num_heads,
            num_layers=moe_layers, num_experts=num_experts)
        # 重新实例化激活被漏掉的 TextClassEmbedder，承接维度 D=256
        self.post_embedder = TextClassEmbedder(embed_dim)

    def forward(self) -> torch.Tensor:
        # 1. 基础 CoOp 分支编码
        T = self.text_branch()
        # 2. 文本专属 Transformer-MoE 模块聚合增强
        T = self.moe_encoder(T.unsqueeze(0)).squeeze(0)
        # 3. 完美激活：通过 LayerNorm + 对比仿射变换头进行特征提纯与规范化 L2 约束
        return self.post_embedder(T)


class EnDecoderModel(nn.Module):

    def __init__(self,
                 num_classes: int = 3,
                 checkpoint_path: str = '/media/tc/7810057410053B20/sy/Medical-SAM-Bench-main/Sam_prepth2.0/sam2_hiera_large.pt',
                 conch_model=None,
                 class_names: list = None,
                 embed_dim: int = 256,
                 eta: list = None):
        super().__init__()
        self.num_classes = num_classes
        self.use_text_align = (conch_model is not None and class_names is not None)

        config_dir = "/media/tc/7810057410053B20/sy/Medical-SAM-Bench-main/sam2/configs/sam2"
        model_cfg = "sam2_hiera_l.yaml"

        model1 = build_sam2(model_cfg, checkpoint_path, config_dir=config_dir)
        model2 = build_sam2(model_cfg, checkpoint_path, config_dir=config_dir)

        del_attrs = ['sam_mask_decoder', 'sam_prompt_encoder', 'memory_encoder',
                     'memory_attention', 'mask_downsample',
                     'obj_ptr_tpos_proj', 'obj_ptr_proj']
        for attr in del_attrs:
            for m in [model1, model2]:
                if hasattr(m, attr):
                    delattr(m, attr)
        for m in [model1, model2]:
            if hasattr(m.image_encoder, 'neck'):
                del m.image_encoder.neck

        self.backbone1 = model1.image_encoder.trunk
        self.backbone2 = model2.image_encoder.trunk
        for p in self.backbone1.parameters(): p.requires_grad = False
        for p in self.backbone2.parameters(): p.requires_grad = False

        self.backbone1.blocks = nn.Sequential(
            *[Adapter(b) for b in self.backbone1.blocks])
        self.backbone2.blocks = nn.Sequential(
            *[Adapter(b) for b in self.backbone2.blocks])

        self.conv1 = nn.Conv2d(2, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 3, 3, padding=1)

        self.mccm1 = OmniScanMoE(dim=144, n_experts=4, top_k=2)
        self.mccm2 = OmniScanMoE(dim=288, n_experts=4, top_k=2)
        self.mccm3 = OmniScanMoE(dim=576, n_experts=4, top_k=2)
        self.mccm4 = OmniScanMoE(dim=1152, n_experts=4, top_k=2)

        self.conv4_fuse = BasicConv2d(144, 144, 1)
        self.conv3_fuse = BasicConv2d(288, 288, 1)
        self.conv2_fuse = BasicConv2d(576, 576, 1)
        self.conv1_fuse = BasicConv2d(1152, 1152, 1)

        self.mlpdecoder = DecoderHead(
            in_channels=[144, 288, 576, 1152],
            num_classes=self.num_classes
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)

        if self.use_text_align:
            assert len(class_names) == num_classes, \
                f"class_names 长度 {len(class_names)} 须等于 num_classes {num_classes}"

            self.text_branch = TextBranchWithMoE(
                conch_model=conch_model, class_names=class_names,
                embed_dim=embed_dim, num_context_tokens=16,
                moe_layers=2, num_heads=8, num_experts=8,
            )
            self.text_align = BrainTumorTextVisualMoE(
                in_channels=[144, 288, 576, 1152],
                embed_dim=embed_dim,
                num_classes=num_classes,
                top_k=2,
                eta=eta,
                load_loss_w=0.01,
                ot_loss_w=0.005,
                hinge_loss_w=0.025,
                margin=0.15
            )


        self.total_mccm = 0.0
        self.total_load = 0.0
        self.total_ot = 0.0
        self.total_hinge = 0.0
        self.batch_count = 0
        self.last_printed_epoch = -1

    def forward(self, x: torch.Tensor, epoch=0, total_epochs=100):
        x1 = self.conv1(torch.cat([x[:, 1:2], x[:, 2:3]], dim=1))
        x2 = self.conv2(torch.cat([x[:, 0:1], x[:, 3:4]], dim=1))

        f1_1, f1_2, f1_3, f1_4 = self.backbone1(x1)
        f2_1, f2_2, f2_3, f2_4 = self.backbone2(x2)

        for _m in [self.mccm1, self.mccm2, self.mccm3, self.mccm4]:
            _m.set_epoch(epoch, total_epochs)
        fuse1, l1 = self.mccm1(f1_1, f2_1)
        fuse2, l2 = self.mccm2(f1_2, f2_2)
        fuse3, l3 = self.mccm3(f1_3, f2_3)
        fuse4, l4 = self.mccm4(f1_4, f2_4)
        loss_mccm = l1 + l2 + l3 + l4

        fuse1 = self.conv4_fuse(fuse1)
        fuse2 = self.conv3_fuse(fuse2)
        fuse3 = self.conv2_fuse(fuse3)
        fuse4 = self.conv1_fuse(fuse4)

        dec_out = self.upsample4(
            self.mlpdecoder([fuse1, fuse2, fuse3, fuse4]))

        if self.use_text_align:
            T = self.text_branch()

            if self.training:
                out, load_loss, ot_intra_loss, hinge_loss = self.text_align(
                    fuse_list=[fuse1, fuse2, fuse3, fuse4],
                    T=T,
                    decoder_out=dec_out,
                    epoch=epoch,
                    total_epochs=total_epochs
                )

                # 累加
                self.total_mccm += loss_mccm.item()
                self.total_load += load_loss.item()
                self.total_ot += ot_intra_loss.item()
                self.total_hinge += hinge_loss.item()
                self.batch_count += 1

                # 每个 epoch 打印一次平均
                if epoch != self.last_printed_epoch:
                    avg_mccm = self.total_mccm / self.batch_count
                    avg_load = self.total_load / self.batch_count
                    avg_ot = self.total_ot / self.batch_count
                    avg_hinge = self.total_hinge / self.batch_count

                    print(f"[Epoch {epoch:2d}] mccm: {avg_mccm:.4f} | load: {avg_load:.4f} | ot: {avg_ot:.4f} | hinge: {avg_hinge:.4f}")

                    # 重置
                    self.last_printed_epoch = epoch
                    self.total_mccm = 0.0
                    self.total_load = 0.0
                    self.total_ot = 0.0
                    self.total_hinge = 0.0
                    self.batch_count = 0

                return out, loss_mccm, load_loss, ot_intra_loss, hinge_loss
            else:
                out = self.text_align(
                    fuse_list=[fuse1, fuse2, fuse3, fuse4],
                    T=T,
                    decoder_out=dec_out,
                    epoch=epoch,
                    total_epochs=total_epochs
                )
        else:
            out = dec_out

        return out


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '/media/tc/7810057410053B20/sy/Medical-SAM-Bench-main/CONCH')
    from CONCH.conch.open_clip_custom import create_model_from_pretrained
    from thop import profile


    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    conch_model, _ = create_model_from_pretrained(
        model_cfg='conch_ViT-B-16',
        checkpoint_path='/media/tc/7810057410053B20/sy/Medical-SAM-Bench-main/CONCH/conch/pytorch_model.bin',
        device=device
    )
    conch_model = conch_model.to(device).eval()

    class_names = [
        "whole tumor in brain MRI",
        "enhancing tumor in brain MRI",
        "tumor core in brain MRI",
    ]

    model = EnDecoderModel(
        num_classes=3,
        checkpoint_path='/media/tc/7810057410053B20/sy/Medical-SAM-Bench-main/Sam_prepth2.0/sam2_hiera_large.pt',
        conch_model=conch_model,
        class_names=class_names,
        embed_dim=256,
        eta=[0.1, 0.2, 0.3, 0.4],
    ).to(device)

    test_input = torch.randn(1, 4, 224, 224).to(device)
    model.train()
    out, loss_mccm, load_loss, ot_loss, hinge_loss = model(test_input, )
    print("========= 修复版全流程前向验证 =========")
    print(f"✅ 分割输出尺寸:                   {out.shape}")
    print(f"✅ 基础融合损失 loss_mccm:         {loss_mccm.item():.4f}")
    print(f"✅ 门控负载均衡 load_loss:          {load_loss.item():.4f}")
    print(f"✅ 宏观对齐损失 ot_intra_loss:      {ot_loss.item():.4f}")
    print(f"✅ 微观判别损失 hinge_loss:         {hinge_loss.item():.4f}")

    print("\n========= 开销度量 =========")
    input_flops = torch.randn(1, 4, 224, 224).to(device)
    flops, params = profile(model, inputs=(input_flops,))
    print(f"✅ FLOPs:           {flops / 1e9:.2f} G")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🔹 总参数量:         {total / 1e6:.2f} M")
    print(f"✅ 可训练参数量:    {trainable / 1e6:.2f} M")
