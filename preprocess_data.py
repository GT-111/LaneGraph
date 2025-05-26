import os
import math
import numpy as np
import cv2
import torch
import json
import pickle
from pathlib import Path
from shapely.geometry import LineString, box, MultiLineString
from roadstructure import LaneMap


def draw_gaussian(heatmap, center, radius=3, value=1):
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                heatmap[ny, nx] = value
    return heatmap


def robust_clip_edge(p1, p2, patch_size, margin):
    line = LineString([p1, p2])
    patch_box = box(margin, margin, patch_size + margin, patch_size + margin)
    if not line.intersects(patch_box):
        return None
    clipped = line.intersection(patch_box)
    if clipped.is_empty:
        return None
    if isinstance(clipped, LineString):
        return list(clipped.coords)
    elif isinstance(clipped, MultiLineString):
        longest = max(clipped.geoms, key=lambda l: l.length)
        return list(longest.coords)
    return None


def resample_polyline(coords, spacing):
    if len(coords) < 2:
        return coords
    line = LineString(coords)
    num_pts = max(int(line.length // spacing), 1)
    return [list(line.interpolate(float(i)/num_pts, normalized=True).coords[0]) for i in range(num_pts + 1)]


def preprocess_all(region_json_path, raw_data_path, processed_data_path, vis_dir,
                    patch_size=640, tile_size=4096, stride=4096, resample_spacing=20, lane_width=5):
    Path(processed_data_path).mkdir(parents=True, exist_ok=True)
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    margin = 128
    regions = json.load(open(region_json_path))
    counter = 0
    counter_out = 0

    for region in regions:
        min_lat, min_lon = region["lat"], region["lon"]
        blocks = [region["ilat"], region["ilon"]]

        for ilat in range(blocks[0]):
            for ilon in range(blocks[1]):
                subregion = [
                    min_lat + ilat * stride / 8 / 111111.0,
                    min_lon + ilon * stride / 8 / 111111.0 / math.cos(math.radians(min_lat)),
                    min_lat + (ilat * stride + tile_size) / 8 / 111111.0,
                    min_lon + (ilon * stride + tile_size) / 8 / 111111.0 / math.cos(math.radians(min_lat))
                ]
                img = cv2.imread(f"{raw_data_path}/sat_{counter}.jpg")
                try:
                    labels = pickle.load(open(f"{raw_data_path}/sat_{counter}_label.p", "rb"))
                except:
                    break

                roadlabel, masklabel = labels
                nodes = roadlabel.nodes
                node_types = roadlabel.nodeType
                edges = roadlabel.neighbors
                edge_types = roadlabel.edgeType

                for sr in range(0, tile_size - patch_size + 1, patch_size // 4):
                    for sc in range(0, tile_size - patch_size + 1, patch_size // 4):
                        sat_crop = img[sr:sr + patch_size, sc:sc + patch_size, :]
                        sat_padded = np.zeros((patch_size + 2 * margin, patch_size + 2 * margin, 3), dtype=np.uint8)
                        sat_padded[margin:margin + patch_size, margin:margin + patch_size] = sat_crop

                        local_nodes = {}
                        local_edges = []
                        node_pos_cache = {}

                        for nid, pos in nodes.items():
                            x, y = pos[0] - sc, pos[1] - sr
                            node_pos_cache[nid] = (x, y)
                            if 0 <= x < patch_size and 0 <= y < patch_size:
                                local_nodes[nid] = {"pos": [x, y], "type": node_types.get(nid, "way")}

                        next_virtual_id = max(nodes.keys(), default=1000000) + 1

                        for nid, neis in edges.items():
                            for nn in neis:
                                if nid not in node_pos_cache or nn not in node_pos_cache:
                                    continue

                                p1 = node_pos_cache[nid]
                                p2 = node_pos_cache[nn]

                                node_type = 'link' if node_types.get(nid) == 'link' or node_types.get(nn) == 'link' else 'way'
                                edge_type = edge_types.get((nid, nn)) or edge_types.get((nn, nid))
                                if edge_type is None:
                                    continue

                                in1 = margin <= p1[0] < patch_size + margin and margin <= p1[1] < patch_size + margin
                                in2 = margin <= p2[0] < patch_size + margin and margin <= p2[1] < patch_size + margin

                                coords = [p1, p2] if in1 and in2 else robust_clip_edge(p1, p2, patch_size, margin)
                                if coords is None or len(coords) < 2:
                                    continue

                                resampled = resample_polyline(coords, resample_spacing)
                                prev_id = None
                                entered_patch = False
                                for i, pt in enumerate(resampled):
                                    within_patch = margin <= pt[0] < patch_size + margin and margin <= pt[1] < patch_size + margin
                                    if within_patch and not entered_patch:
                                        prev_id = nid
                                        entered_patch = True
                                        local_nodes[prev_id] = {"pos": list(pt), "type": node_type}
                                        continue
                                    if entered_patch:
                                        curr_id = next_virtual_id
                                        local_nodes[curr_id] = {"pos": list(pt), "type": node_type}
                                        next_virtual_id += 1
                                        local_edges.append({"from": prev_id, "to": curr_id, "type": edge_type})
                                        prev_id = curr_id

                        # prepare training targets
                        H = W = patch_size + 2 * margin
                        img_tensor = torch.from_numpy(sat_padded).permute(2, 0, 1).float() / 255.0

                        node_heatmap = torch.zeros((1, H, W))
                        node_type = torch.zeros((2, H, W))
                        edge_vector = torch.zeros((2, H, W))
                        edge_prob = torch.zeros((1, H, W))
                        edge_class = torch.zeros((3, H, W))
                        segmentation = torch.zeros((1, H, W))
                        topo_tensor = torch.zeros((7, H, W))

                        for nid, node in local_nodes.items():
                            x, y = map(int, node["pos"])
                            draw_gaussian(node_heatmap[0], (x, y), radius=2)
                            node_type[0 if node["type"] == "way" else 1, y, x] = 1
                            disk = np.zeros((H, W), dtype=np.uint8)
                            cv2.circle(disk, (x, y), radius=4, color=1, thickness=-1)
                            segmentation[0] += torch.from_numpy(disk)
                            topo_tensor[1, y, x] = 1

                        for edge in local_edges:
                            x1, y1 = local_nodes[edge["from"]]["pos"]
                            x2, y2 = local_nodes[edge["to"]]["pos"]
                            edge_type_index = 0 if edge["type"] == "way" else 1

                            dx = x2 - x1
                            dy = y2 - y1
                            length = math.hypot(dx, dy) + 1e-5
                            dx_norm = dx / length
                            dy_norm = dy / length

                            mask = np.zeros((H, W), dtype=np.uint8)
                            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), color=1, thickness=lane_width)
                            ys, xs = np.where(mask > 0)
                            for x, y in zip(xs, ys):
                                segmentation[0, y, x] = 1
                                edge_prob[0, y, x] = 1
                                edge_class[edge_type_index, y, x] = 1
                                topo_tensor[0, y, x] = 1
                                topo_tensor[2, y, x] = 1
                                edge_vector[:, y, x] = torch.tensor([dx_norm, dy_norm])
                                topo_tensor[3, y, x] = dx_norm
                                topo_tensor[4, y, x] = dy_norm

                        np.save(os.path.join(processed_data_path, f"sample_{counter_out:05d}.npy"), {
                            "global_id": f"{counter_out}",
                            "region": subregion,
                            "nodes": local_nodes,
                            "edges": local_edges,
                            "image": img_tensor.numpy(),
                            "node_heatmap": node_heatmap.numpy(),
                            "node_type": node_type.numpy(),
                            "edge_vector": edge_vector.numpy(),
                            "edge_prob": edge_prob.numpy(),
                            "edge_class": edge_class.numpy(),
                            "segmentation": segmentation.numpy(),
                            "topo_tensor": topo_tensor.numpy()
                        }, allow_pickle=True)
                        counter_out += 1
                

    print("✅ Preprocessing complete. Total samples:", counter_out)


if __name__ == "__main__":
    raw_data_path = '/mnt/c/Users/hg25079/Documents/GitHub/LaneGraph/raw_data'
    processed_data_path = '/mnt/c/Users/hg25079/Documents/GitHub/LaneGraph/processed_data'
    vis_dir = '/mnt/c/Users/hg25079/Documents/GitHub/LaneGraph/visualizations'
    region_json_path = '/mnt/c/Users/hg25079/Documents/GitHub/LaneGraph/raw_data/regions.json'
    preprocess_all(
        region_json_path=region_json_path,
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        vis_dir=vis_dir
    )
