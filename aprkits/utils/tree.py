from aprkits.types import TreePathData


def give_node_pos_coeffs(tree: TreePathData):
    nodes = tree['nodes']
    nodes[0]['coeff'] = 0.5

    for node in nodes:
        children = node['children']
        c = len(children)
        for i, child in enumerate(children):
            coeff = (c - i) / (c + 1.0)
            nodes[child]['coeff'] = coeff

    return tree
