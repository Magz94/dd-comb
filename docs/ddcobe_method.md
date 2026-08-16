# Dempster–Dirichlet Combination of Backbone Experts (DD-CoBE)

## Method

### Overview and notation

Let $x$ be an input image and let $\{\phi_m\}_{m=1}^{K}$ be $K$ heterogeneous,
**frozen** ImageNet-pretrained backbones (here $K=5$: ResNet-18, VGG-16,
DenseNet-121, EfficientNet-B0, ViT-B/16). For a $C$-class target task we treat
each backbone as a *view* of the input and combine the views at the level of
**evidence** rather than at the feature or logit level. The backbones are never
updated; only the lightweight per-view heads, the fusion path, and the
combination are trained on the few-shot target data.

| Symbol | Meaning |
|---|---|
| $z_m \in \mathbb{R}^{D_m}$ | frozen embedding of backbone $m$ |
| $t_m \in \mathbb{R}^{H}$ | adapted, normalized token of view $m$ |
| $\ell_v \in \mathbb{R}^{C}$ | logits of view $v$ |
| $e_v, \alpha_v \in \mathbb{R}^{C}_{\ge 0}$ | evidence and Dirichlet concentration of view $v$ |
| $(\mathbf{b}_v, u_v)$ | subjective opinion (belief masses, uncertainty mass) of view $v$ |
| $\alpha^{\oplus}$ | concentration of the Dempster-combined opinion |
| $\psi(\cdot)$ | digamma function |

### 1. Frozen backbones and per-backbone experts

Each frozen embedding is projected into a shared $H$-dimensional space by a
per-view MLP adapter and layer-normalized,
$$
g_m = \mathbf{W}^{(2)}_m\,\rho\!\big(\mathbf{W}^{(1)}_m z_m\big), 
\qquad 
t_m = \mathrm{LayerNorm}(g_m), 
\qquad m = 1,\dots,K,
$$
where $\rho$ is the ReLU nonlinearity. A linear **expert head** maps each token
to class logits,
$$
\ell_m = \mathbf{W}_m\, t_m + \mathbf{b}_m \;\in\; \mathbb{R}^{C}.
$$
The pair *(adapter + linear head)* constitutes the head of view $m$; together
the $K$ experts form an ensemble of backbone-specific classifiers.

**Fused view (optional).** In parallel, the normalized tokens
$T = [t_1,\dots,t_K]$ are processed by the Dirichlet-attention fusion path of
the base architecture — a Dirichlet multi-head self-attention block, an
attention-pooling layer and a projection head — yielding a single fused
embedding $f$ and its logits
$$
\ell_f = \mathbf{W}_{\!f}\, f + \mathbf{b}_{\!f} \;\in\; \mathbb{R}^{C}.
$$
The fused view is treated as one additional opinion. We denote the set of views
by $\mathcal{V} = \{1,\dots,K\}\cup\{f\}$ (without the fused view,
$\mathcal{V}=\{1,\dots,K\}$).

### 2. Subjective-logic opinions

Following evidential deep learning and subjective logic, each view's logits are
read as non-negative **evidence** and mapped to a Dirichlet distribution over
the class simplex,
$$
e_v = \mathrm{softplus}(\ell_v),
\qquad
\alpha_v = e_v + \mathbf{1},
\qquad
S_v = \sum_{k=1}^{C}\alpha_{v,k},
$$
where $S_v$ is the Dirichlet *strength*. The opinion of view $v$ is the pair of
per-class belief masses $\mathbf{b}_v$ and an uncertainty (vacuity) mass $u_v$,
$$
b_{v,k} = \frac{e_{v,k}}{S_v} = \frac{\alpha_{v,k}-1}{S_v},
\qquad
u_v = \frac{C}{S_v},
\qquad
u_v + \sum_{k=1}^{C} b_{v,k} = 1 .
$$
A view that observes little evidence has $S_v \!\to\! C$ and $u_v \!\to\! 1$
(a vacuous opinion); a confident view has $u_v \!\to\! 0$. The Dirichlet mean of
a single view is $\hat p_{v,k} = \alpha_{v,k}/S_v$.

### 3. Dempster–Shafer combination

The views are fused with the reduced Dempster rule of combination. For two
opinions $\mathcal{M}^{1}=(\mathbf b^{1},u^{1})$ and
$\mathcal{M}^{2}=(\mathbf b^{2},u^{2})$, the joint opinion
$\mathcal{M}=\mathcal{M}^{1}\oplus\mathcal{M}^{2}=(\mathbf b,u)$ is
$$
b_k = \frac{1}{1-\kappa}\big(b^{1}_k b^{2}_k + b^{1}_k u^{2} + b^{2}_k u^{1}\big),
\qquad
u = \frac{1}{1-\kappa}\, u^{1} u^{2},
$$
$$
\kappa = \sum_{i\neq j} b^{1}_i\, b^{2}_j ,
$$
where $\kappa$ measures the conflict between the two opinions and $1/(1-\kappa)$
renormalizes the non-conflicting mass. The rule is commutative and associative,
so all views are combined by the left fold
$$
\mathcal{M}^{\oplus} \;=\; \bigoplus_{v\in\mathcal{V}} \mathcal{M}_v .
$$
The joint opinion is mapped back to a Dirichlet via the inverse of the
subjective-logic transform,
$$
S^{\oplus} = \frac{C}{u^{\oplus}},
\qquad
e^{\oplus}_k = b^{\oplus}_k \, S^{\oplus},
\qquad
\alpha^{\oplus}_k = e^{\oplus}_k + 1 .
$$

**Parameter-free sample-adaptive weighting.** Because a vacuous opinion
($u_v\!\to\!1,\ \mathbf b_v\!\to\!\mathbf 0$) is the identity element of $\oplus$,
a view that is uncertain on a given input is automatically discounted on that
input, while confident and mutually-agreeing views reinforce one another
($S^{\oplus}$ grows) and conflicting views inflate the joint uncertainty
($u^{\oplus}$ grows). The combination therefore performs a **per-sample,
uncertainty-driven weighting of the experts with no learned gating parameters.**

### 4. Prediction and uncertainty

The predictive distribution is the mean of the combined Dirichlet,
$$
p_k = \frac{\alpha^{\oplus}_k}{S^{\oplus}},
\qquad
\hat y = \arg\max_k p_k ,
$$
and the model additionally exposes an explicit epistemic signal, the combined
vacuity $u^{\oplus}=C/S^{\oplus}$, which a softmax classifier cannot represent.
First-order (total) predictive uncertainty is the entropy $H[\,p\,]$ of the
combined mean.

### 5. Training objective (DD-CoBE *pure*)

Each Dirichlet is trained with the evidential objective of Trusted Multi-View
Classification: the expected cross-entropy under the Dirichlet plus an annealed
wrong-evidence regularizer,
$$
\mathcal{L}_{\text{edl}}(\alpha, y)
=
\underbrace{\sum_{k=1}^{C} y_k\big(\psi(S)-\psi(\alpha_k)\big)}_{\text{expected cross-entropy}}
\;+\;
\lambda(t)\,
\underbrace{\mathrm{KL}\!\big[\mathrm{Dir}(\tilde{\alpha})\,\big\|\,\mathrm{Dir}(\mathbf 1)\big]}_{\text{wrong-evidence penalty}},
$$
where $\tilde{\alpha} = y + (1-y)\odot\alpha$ masks the evidence of the true
class so that only non-target evidence is penalized toward the uniform
Dirichlet, and $\lambda(t) = \lambda_{0}\,\min\!\big(1, t/T_{\text{w}}\big)$
anneals the KL term over the first $T_{\text{w}}$ epochs. The KL term has the
closed form
$$
\mathrm{KL}\!\big[\mathrm{Dir}(\tilde\alpha)\,\|\,\mathrm{Dir}(\mathbf 1)\big]
=
\log\frac{\Gamma\!\big(\textstyle\sum_k\tilde\alpha_k\big)}{\Gamma(C)\prod_k\Gamma(\tilde\alpha_k)}
+
\sum_{k=1}^{C}(\tilde\alpha_k-1)\Big[\psi(\tilde\alpha_k)-\psi\!\big(\textstyle\sum_j\tilde\alpha_j\big)\Big].
$$

The *pure* objective applies $\mathcal{L}_{\text{edl}}$ to the combined opinion
and, as deep supervision, to every per-backbone expert, together with a
head-diversity regularizer $\mathcal{L}_{\text{div}}$ on the attention heads:
$$
\boxed{\;
\mathcal{L}
=
\mathcal{L}_{\text{edl}}\!\big(\alpha^{\oplus}, y\big)
\;+\;
\frac{1}{K}\sum_{m=1}^{K}\mathcal{L}_{\text{edl}}\!\big(\alpha_m, y\big)
\;+\;
\lambda_{\text{div}}\,\mathcal{L}_{\text{div}}
\;}
$$
No cross-entropy on a softmax head is used: the combined Dirichlet mean is
optimized directly. The combined term shapes the final decision and propagates
gradients through $\oplus$ into all views, while the per-expert terms guarantee
that every backbone receives full supervision regardless of its current
influence in the combination and that its evidence is calibrated so that
$u_m$ reflects the reliability of view $m$.

### Remarks

DD-CoBE replaces the *learned, weighted* combination of a mixture-of-experts
(additive logit mixing with a gating router) by an *uncertainty-driven*
combination in opinion space (multiplicative Dempster fusion). The combination
adds **no parameters** and requires **no router**, which is its key property in
the few-shot regime, where per-sample combination weights cannot be reliably
estimated. The Dirichlet-attention path fuses information at the *feature*
level and the Dempster rule fuses it at the *decision* level, so uncertainty is
modeled consistently end to end.
