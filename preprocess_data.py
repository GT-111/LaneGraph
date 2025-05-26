import json
import math
import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from shapely.geometry import LineString, box, MultiLineString
from roadstructure import LaneMap

project_path = '/home/hetianguo/Desktop/LaneGraph/'
raw_data_path = os.path.join(project_path, './dataset/')
processed_data_path = os.path.join(project_path, './dataset_processed/')
vis_dir = os.path.join(processed_data_path, 'vis')
Path(processed_data_path).mkdir(parents=True, exist_ok=True)
Path(vis_dir).mkdir(parents=True, exist_ok=True)

MARGIN = 128
TILE_SIZE = 4096
STRIDE = 4096
PATCH_SIZE = 640
RESAMPLE_SPACING = 20  # pixels between resampled points

regions = json.load(open(os.path.join(raw_data_path, 'regions.json')))
counter = 0
counter_out = 0
total_length = 0


def robust_clip_edge(p1, p2, patch_size):
    line = LineString([p1, p2])
    patch_box = box(0, 0, patch_size, patch_size)
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

for region in regions:
    min_lat, min_lon = region["lat"], region["lon"]
    region_tag = region["tag"]
    blocks = [region["ilat"], region["ilon"]]

    for ilat in range(blocks[0]):
        for ilon in range(blocks[1]):
            subregion = [
                min_lat + ilat * STRIDE / 8 / 111111.0,
                min_lon + ilon * STRIDE / 8 / 111111.0 / math.cos(math.radians(min_lat)),
                min_lat + (ilat * STRIDE + TILE_SIZE) / 8 / 111111.0,
                min_lon + (ilon * STRIDE + TILE_SIZE) / 8 / 111111.0 / math.cos(math.radians(min_lat))
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

            for sr in range(0, TILE_SIZE - PATCH_SIZE + 1, PATCH_SIZE//4):
                for sc in range(0, TILE_SIZE - PATCH_SIZE + 1, PATCH_SIZE//4):
                    sat_crop = img[sr:sr + PATCH_SIZE, sc:sc + PATCH_SIZE, :]
                    sat_padded = np.zeros((PATCH_SIZE + 2 * MARGIN, PATCH_SIZE + 2 * MARGIN, 3), dtype=np.uint8)
                    sat_padded[MARGIN:MARGIN + PATCH_SIZE, MARGIN:MARGIN + PATCH_SIZE] = sat_crop
                    vis_img = sat_padded.copy()

                    local_nodes = {}
                    local_edges = []
                    node_pos_cache = {}

                    for nid, pos in nodes.items():
                        x, y = pos[0] - sc, pos[1] - sr
                        node_pos_cache[nid] = (x, y)
                        if 0 <= x < PATCH_SIZE and 0 <= y < PATCH_SIZE:
                            local_nodes[nid] = {"pos": [x, y], "type": node_types.get(nid, "way")}

                    next_virtual_id = max(nodes.keys(), default=1000000) + 1

                    for nid, neis in edges.items():
                        for nn in neis:
                            if nid not in node_pos_cache or nn not in node_pos_cache:
                                continue

                            p1 = node_pos_cache[nid]
                            p2 = node_pos_cache[nn]
                            edge_type = edge_types.get((nid, nn)) or edge_types.get((nn, nid))
                            if edge_type is None:
                                continue

                            in1 = 0 <= p1[0] < PATCH_SIZE and 0 <= p1[1] < PATCH_SIZE
                            in2 = 0 <= p2[0] < PATCH_SIZE and 0 <= p2[1] < PATCH_SIZE

                            if in1 and in2:
                                coords = [p1, p2]
                            else:
                                coords = robust_clip_edge(p1, p2, PATCH_SIZE)
                            if coords is None or len(coords) < 2:
                                continue

                            resampled = resample_polyline(coords, RESAMPLE_SPACING)
                            prev_id = None
                            for i, pt in enumerate(resampled):
                                if i == 0 and in1:
                                    prev_id = nid
                                    continue
                                if i == 0 and not in1:
                                    prev_id = next_virtual_id
                                    local_nodes[prev_id] = {"pos": list(pt), "type": "virtual"}
                                    next_virtual_id += 1
                                    continue
                                curr_id = next_virtual_id
                                local_nodes[curr_id] = {"pos": list(pt), "type": "virtual"}
                                next_virtual_id += 1
                                local_edges.append({"from": prev_id, "to": curr_id, "type": edge_type, "dir": None})
                                prev_id = curr_id

                    for edge in local_edges:
                        x1, y1 = local_nodes[edge["from"]]["pos"]
                        x2, y2 = local_nodes[edge["to"]]["pos"]
                        dx = x2 - x1
                        dy = y2 - y1
                        l = math.sqrt(dx * dx + dy * dy) + 1e-5
                        edge["dir"] = [dx / l, dy / l]
                        color = (0, 255, 0) if edge["type"] == "way" else (0, 0, 255)
                        cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    for nid, node in local_nodes.items():
                        x, y = node["pos"]
                        color = (255, 0, 0) if node["type"] == "way" else (0, 255, 255)
                        cv2.circle(vis_img, (int(x), int(y)), 3, color, -1)

                    MIN_NODES = 50
                    node_cnt_in_sat = sum(
                        1
                        for v in local_nodes.values()
                        if MARGIN <= v["pos"][0] < MARGIN + PATCH_SIZE
                        and MARGIN <= v["pos"][1] < MARGIN + PATCH_SIZE
                    )

                    if node_cnt_in_sat < MIN_NODES:
                        continue  # skip low-quality samples

                    sample = {
                        "global_id": f"{ilat}_{ilon}_{counter_out}",
                        "image": sat_padded,
                        "nodes": local_nodes,
                        "edges": local_edges,
                        "region": subregion,
                    }

                    np.save(os.path.join(processed_data_path, f"sample_{counter_out}.npy"), sample, allow_pickle=True)
                    cv2.imwrite(os.path.join(vis_dir, f"vis_{counter_out}.jpg"), vis_img)
                    counter_out += 1

            counter += 1

print("Total samples:", counter_out)