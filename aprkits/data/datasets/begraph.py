from torch.utils.data import Dataset


class BatchEncodingGraphDataset(Dataset):
    """
    dataset parts:
      - tokens: BatchEncoding
      - graphNodes: BatchEncoding
      - graphTypeNodes: BatchEncoding
      - graphNodeChildCounts: BatchEncoding
    """

    def __init__(self, inputs, graph_nodes, graph_type_nodes, graph_child_counts, targets):
        assert len(inputs['input_ids']) == len(targets['input_ids'])
        assert len(graph_nodes['input_ids']) == len(inputs['input_ids'])
        assert len(graph_type_nodes['input_ids']) == len(inputs['input_ids'])
        assert len(graph_child_counts) == len(inputs['input_ids'])
        assert len(inputs['attention_mask']) == len(targets['attention_mask'])

        self.inp_data = inputs['input_ids']
        self.inp_graph_data = graph_nodes['input_ids']
        self.inp_graph_type_data = graph_type_nodes['input_ids']
        self.inp_graph_child_counts = graph_child_counts
        self.inp_data_mask = inputs['attention_mask']
        self.inp_graph_data_mask = graph_nodes['attention_mask']
        self.tar_data = targets['input_ids']
        self.tar_data_mask = targets['attention_mask']

    def __getitem__(self, index):
        return (
            self.inp_data[index], self.inp_graph_data[index],
            self.inp_graph_type_data[index], self.inp_graph_child_counts[index],
            self.inp_data_mask[index], self.inp_graph_data_mask[index],
            self.tar_data[index], self.tar_data_mask[index]
        )

    def __len__(self):
        return len(self.inp_data)
