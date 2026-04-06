"""Convert pymatgen Structure objects into MatGL-compatible graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matgl.ext._pymatgen_pyg import Structure2Graph, get_element_list

if TYPE_CHECKING:
    from pymatgen.core import Structure


def get_converter(
    structures: list[Structure],
    cutoff: float = 5.0,
    element_types: tuple[str, ...] | None = None,
) -> Structure2Graph:
    """Build a Structure2Graph converter from a list of structures.

    If *element_types* is ``None``, it is inferred from the structures.
    """
    if element_types is None:
        element_types = get_element_list(structures)
    return Structure2Graph(element_types=element_types, cutoff=cutoff)


def structure_to_graph(
    structure: Structure,
    converter: Structure2Graph,
):
    """Convert a single pymatgen Structure to a PyG Data graph.

    Returns ``(graph, lattice_tensor, state_attr)``.
    """
    return converter.get_graph(structure)
