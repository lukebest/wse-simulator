"""Global color assignment for TDM flattened butterfly overlays."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


LogicalLink = tuple[int, int]
PhysicalLink = tuple[int, int]


@dataclass(slots=True)
class ColorPlan:
    C: int
    link_active: dict[PhysicalLink, list[LogicalLink | None]]
    color_of_logical: dict[LogicalLink, int]
    logical_dims: dict[LogicalLink, int]
    load_per_physical_link: dict[PhysicalLink, int]
    color_lower_bound: int


def _build_link_active(
    color_of_logical: dict[LogicalLink, int],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    color_count: int,
) -> dict[PhysicalLink, list[LogicalLink | None]]:
    link_active: dict[PhysicalLink, list[LogicalLink | None]] = defaultdict(
        lambda: [None] * color_count
    )
    for logical_link, color in color_of_logical.items():
        for edge in physical_paths.get(logical_link, []):
            if link_active[edge][color] is not None and link_active[edge][color] != logical_link:
                raise ValueError("Color conflict: two logical links share one physical edge in same color.")
            link_active[edge][color] = logical_link
    return dict(link_active)


def assign_colors(
    logical_links: list[tuple[int, int, int]],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
) -> ColorPlan:
    """Assign one global color per logical link with physical-link conflict avoidance."""
    if not logical_links:
        return ColorPlan(
            C=1,
            link_active={edge: [None] for edge in physical_links},
            color_of_logical={},
            logical_dims={},
            load_per_physical_link={edge: 0 for edge in physical_links},
            color_lower_bound=1,
        )

    logical_with_dim: list[tuple[LogicalLink, int]] = [((u, v), dim) for u, v, dim in logical_links]
    load_per_link: dict[PhysicalLink, int] = {edge: 0 for edge in physical_links}
    physical_to_logicals: dict[PhysicalLink, list[LogicalLink]] = defaultdict(list)

    for logical_link, _ in logical_with_dim:
        path = physical_paths.get(logical_link, [])
        for edge in path:
            load_per_link[edge] = load_per_link.get(edge, 0) + 1
            physical_to_logicals[edge].append(logical_link)

    lower_bound = max([1, *load_per_link.values()])
    color_count = lower_bound
    color_of_logical: dict[LogicalLink, int] = {}
    logical_dims: dict[LogicalLink, int] = {}
    free_at: dict[PhysicalLink, list[bool]] = {
        edge: [True] * color_count for edge in load_per_link
    }

    # Longest-path-first ordering improves greedy packing quality.
    ordered = sorted(logical_with_dim, key=lambda item: -len(physical_paths.get(item[0], [])))
    for logical_link, dim in ordered:
        path = physical_paths.get(logical_link, [])
        placed = False
        for color in range(color_count):
            if all(free_at.setdefault(edge, [True] * color_count)[color] for edge in path):
                color_of_logical[logical_link] = color
                logical_dims[logical_link] = dim
                for edge in path:
                    free_at[edge][color] = False
                placed = True
                break
        if placed:
            continue

        for edge in free_at:
            free_at[edge].append(True)
        color_count += 1
        new_color = color_count - 1
        color_of_logical[logical_link] = new_color
        logical_dims[logical_link] = dim
        for edge in path:
            free_at.setdefault(edge, [True] * color_count)
            free_at[edge][new_color] = False

    link_active = _build_link_active(color_of_logical, physical_paths, color_count)

    return ColorPlan(
        C=color_count,
        link_active=link_active,
        color_of_logical=color_of_logical,
        logical_dims=logical_dims,
        load_per_physical_link=load_per_link,
        color_lower_bound=lower_bound,
    )


def assign_colors_for_routes(
    logical_links: list[tuple[int, int, int]],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
    force_monochrome_paths: list[list[LogicalLink]],
) -> ColorPlan:
    """Greedy coloring with per-route monochrome constraints.

    Each path in *force_monochrome_paths* must use one constant color across
    all of its hops.  Only hops on the same path are equalized (no global
    transitive closure across different routes).
    """
    if not logical_links:
        return ColorPlan(
            C=1,
            link_active={edge: [None] for edge in physical_links},
            color_of_logical={},
            logical_dims={},
            load_per_physical_link={edge: 0 for edge in physical_links},
            color_lower_bound=1,
        )

    logical_with_dim: list[tuple[LogicalLink, int]] = [((u, v), dim) for u, v, dim in logical_links]
    all_logical = [link for link, _ in logical_with_dim]
    path_mates: dict[LogicalLink, set[LogicalLink]] = {link: set() for link in all_logical}
    for path in force_monochrome_paths:
        for link in path:
            path_mates.setdefault(link, set()).update(h for h in path if h != link)

    load_per_link: dict[PhysicalLink, int] = {edge: 0 for edge in physical_links}
    for logical_link, _ in logical_with_dim:
        for edge in physical_paths.get(logical_link, []):
            load_per_link[edge] = load_per_link.get(edge, 0) + 1

    lower_bound = max([1, *load_per_link.values()])
    color_count = lower_bound
    color_of_logical: dict[LogicalLink, int] = {}
    logical_dims: dict[LogicalLink, int] = {link: dim for link, dim in logical_with_dim}
    free_at: dict[PhysicalLink, list[bool]] = {
        edge: [True] * color_count for edge in load_per_link
    }

    def required_color(link: LogicalLink) -> int | None:
        mates = [color_of_logical[m] for m in path_mates.get(link, ()) if m in color_of_logical]
        if not mates:
            return -1
        if len(set(mates)) > 1:
            return None
        return mates[0]

    def try_place(count: int) -> bool:
        free_at.clear()
        for edge in load_per_link:
            free_at[edge] = [True] * count
        color_of_logical.clear()

        path_first = {link for path in force_monochrome_paths for link in path}
        ordered = sorted(
            logical_with_dim,
            key=lambda item: (
                item[0] not in path_first,
                -len(physical_paths.get(item[0], [])),
            ),
        )
        for logical_link, dim in ordered:
            path = physical_paths.get(logical_link, [])
            req = required_color(logical_link)
            if req is None:
                return False
            candidates = [req] if req >= 0 else range(count)
            placed = False
            for color in candidates:
                if all(free_at.setdefault(edge, [True] * count)[color] for edge in path):
                    color_of_logical[logical_link] = color
                    for edge in path:
                        free_at[edge][color] = False
                    placed = True
                    break
            if not placed:
                return False
        return True

    while not try_place(color_count):
        color_count += 1

    link_active = _build_link_active(color_of_logical, physical_paths, color_count)
    return ColorPlan(
        C=color_count,
        link_active=link_active,
        color_of_logical=color_of_logical,
        logical_dims=logical_dims,
        load_per_physical_link=load_per_link,
        color_lower_bound=lower_bound,
    )


def route_color(plan: ColorPlan, route_path: list[LogicalLink]) -> int | None:
    """Return the single color for a dim-order route, or None if hops disagree."""
    if not route_path:
        return None
    colors = {plan.color_of_logical[link] for link in route_path}
    if len(colors) != 1:
        return None
    return next(iter(colors))


def min_colors_for_route(
    logical_links: list[tuple[int, int, int]],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
    route_path: list[LogicalLink],
) -> tuple[int, int | None]:
    """Minimum C and chosen color for one src->dst route (monochrome constraint)."""
    if not route_path:
        return 1, None
    plan = assign_colors_for_routes(
        logical_links,
        physical_paths,
        physical_links,
        force_monochrome_paths=[route_path],
    )
    return plan.C, route_color(plan, route_path)
