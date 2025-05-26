from dataset.lane_graph_dataset import LaneGraphDataset
from roadstructure import LaneMap
import os
import pickle

data_dir = '/mnt/c/Users/hg25079/Documents/GitHub/LaneGraph/raw_data'
label_files = [f for f in os.listdir(data_dir) if f.endswith('_label.p')]

for label_file in label_files:
    try:
        print(label_file)
        with open(os.path.join(data_dir, label_file), 'rb') as f:
            labels = pickle.load(f)
            roadlabel, masklabel = labels
            if not isinstance(roadlabel, LaneMap):
                new_roadlabel = LaneMap()
                new_roadlabel.nodes = roadlabel.nodes
                new_roadlabel.nodeType = roadlabel.nodeType
                new_roadlabel.neighbors = roadlabel.neighbors
                new_roadlabel.edgeType = roadlabel.edgeType
                roadlabel = new_roadlabel
            
            if not isinstance(masklabel, LaneMap):
                new_masklabel = LaneMap()
                new_masklabel.nodes = masklabel.nodes
                new_masklabel.nodeType = masklabel.nodeType
                new_masklabel.neighbors = masklabel.neighbors
                new_masklabel.edgeType = masklabel.edgeType
                masklabel = new_masklabel
            # save the new label to another file
            new_label_file = label_file.replace('_label.p', '_new_label.p')
            with open(os.path.join(data_dir, new_label_file), 'wb') as f:
                pickle.dump((roadlabel, masklabel), f)
    except Exception as e:
        print(f"Error processing {label_file}: {str(e)}")
