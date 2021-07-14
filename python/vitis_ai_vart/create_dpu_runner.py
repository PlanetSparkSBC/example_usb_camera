import runner
import xir
from typing import List


def get_child_subgraph_dpu(model_graph: "Graph") -> List["Subgraph"]:
    assert model_graph is not None, "'graph' should not be None."
    root_subgraph = model_graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def create_dpu_runner(model: "xmodel") -> object:
    model_graph = xir.Graph.deserialize(model);
    subgraphs = get_child_subgraph_dpu(model_graph)
    assert len(subgraphs) == 1  #only one DPU kernel
    return subgraphs[0]
