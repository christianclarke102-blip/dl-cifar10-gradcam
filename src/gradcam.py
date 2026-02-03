import torch
import torch.nn.functional as F

class GradCAM:
    """
    Minimal Grad-CAM for CNN models.
    Usage:
      cam = GradCAM(model, target_layer)
      heatmap = cam(x, class_idx)  # returns HxW tensor in [0,1]
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # activations/grads: [B, C, H, W]
        acts = self.activations
        grads = self.gradients

        # global-average-pool gradients -> weights: [B, C, 1, 1]
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # weighted sum across channels -> [B, 1, H, W]
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # normalize to [0,1]
        cam = cam.squeeze(0).squeeze(0)  # [H, W]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
