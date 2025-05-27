import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def lanegraph_collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys if isinstance(batch[0][k], torch.Tensor)}


class LaneGraphDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(path, allow_pickle=True).item()

        return {
            "image": torch.from_numpy(data["image"]).float(),
            "node_heatmap": torch.from_numpy(data["node_heatmap"]).float(),
            "node_type": torch.from_numpy(data["node_type"]).float(),
            "edge_vector": torch.from_numpy(data["edge_vector"]).float(),
            "edge_prob": torch.from_numpy(data["edge_prob"]).float(),
            "edge_class": torch.from_numpy(data["edge_class"]).float(),
            "segmentation": torch.from_numpy(data["segmentation"]).float(),
            "topo_tensor": torch.from_numpy(data["topo_tensor"]).float(),
            "nodes": data["nodes"],
            "edges": data["edges"],
            "region": data["region"],
            "global_id": data["global_id"]
        }

    def visualize_sample(self, idx=None, sample=None, save_path=None, show=True, verbose=True, skip=1):
        if sample is None:
            sample = self[idx if idx is not None else np.random.randint(len(self))]

        img = TF.to_pil_image(sample["image"])
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        node_mask = sample["node_heatmap"][0].numpy()
        edge_prob = sample["edge_prob"][0].numpy()
        segmentation = sample["segmentation"][0].numpy()
        edge_vector = sample["edge_vector"].numpy()

        num_nodes = (node_mask > 0.5).sum()
        num_edges = (edge_prob > 0.5).sum()
        if verbose:
            print(f"🟠 Visualizing sample: {sample['global_id']}")
            print(f"🔴 Total nodes: {num_nodes}")
            print(f"🔵 Total edge pixels: {num_edges}")
            print(f"📌 Region: {sample['region']}")

        seg_mask = np.zeros((H, W, 3), dtype=np.uint8)
        seg_mask[segmentation > 0.5] = [0, 255, 0]

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # Panel 1: RGB image overlay
        ax = axes[0]
        ax.imshow(img_np)
        ax.set_title("Image + Nodes + Directions + Segmentation")
        ys, xs = np.where(node_mask > 0.5)
        ax.scatter(xs, ys, s=10, c='red', label='nodes')
        ys, xs = np.where(edge_prob > 0.5)
        for x, y in zip(xs[::skip], ys[::skip]):
            dx, dy = edge_vector[0, y, x].item(), edge_vector[1, y, x].item()
            ax.arrow(x, y, dx * 6, dy * 6, head_width=2, color='blue', alpha=0.6)
        ax.imshow(seg_mask, alpha=0.3)
        ax.axis('off')

        # Panel 2: segmentation mask
        ax = axes[1]
        ax.imshow(seg_mask)
        ax.set_title("Segmentation Mask")
        ax.axis('off')

        # Panel 3: Quiver plot
        ax = axes[2]
        ax.set_title("Edge Vector Field (Quiver)")
        Y, X = np.meshgrid(np.arange(0, H, skip), np.arange(0, W, skip), indexing='ij')
        U = edge_vector[0, 0:H:skip, 0:W:skip]
        V = edge_vector[1, 0:H:skip, 0:W:skip]
        mask = edge_prob[0:H:skip, 0:W:skip] > 0.5
        ax.quiver(X[mask], Y[mask], U[mask], V[mask], angles='xy', scale_units='xy', scale=1, color='blue')
        ax.imshow(img_np, alpha=0.5)
        ax.axis('off')

        # Panel 4: RGB-encoded direction
        ax = axes[3]
        ax.set_title("RGB Encoded Direction")
        normal_img = np.zeros((H, W, 3), dtype=np.uint8)
        ys, xs = np.where(edge_prob > 0.5)
        for x, y in zip(xs, ys):
            dx, dy = edge_vector[:, y, x]
            r = int(127 + dx * 127)
            g = int(127 + dy * 127)
            normal_img[y, x] = (r, g, 127)
        ax.imshow(normal_img)
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")
        if show:
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    vis_dir = '/mnt/c/Users/hg25079/Documents/GitHub/LaneGraph/visualizations'
    dataset = LaneGraphDataset("processed_data")
    idx = 8
    dataset.visualize_sample(idx=idx, save_path=os.path.join(vis_dir, f"sample_{idx}.png"))
