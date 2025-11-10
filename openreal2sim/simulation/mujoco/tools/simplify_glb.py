#!/usr/bin/env python3
"""Simplify GLB/GLTF mesh files using quadric edge collapse decimation."""

from pathlib import Path
import tempfile
import pymeshlab
import trimesh
import tyro
from loguru import logger


def simplify_glb(
    input_path: Path,
    output_path: Path,
    target_tris: int,
) -> None:
    """Simplify a GLB/GLTF mesh file.
    
    Args:
        input_path: Input GLB/GLTF file path
        output_path: Output GLB/GLTF file path
        target_tris: Target number of triangles
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_obj = tmp_dir / 'mesh.obj'

        logger.info(f"Loading {input_path.name}...")
        scene = trimesh.load(str(input_path), force='scene')
        scene.export(str(tmp_obj), file_type='obj', include_texture=True)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(tmp_obj))
        n_tris = ms.current_mesh().face_number()
        
        logger.info(f"Original: {n_tris:,} triangles")

        tgt = min(int(target_tris), n_tris - 1)
        if tgt >= n_tris:
            logger.info(f"Target {target_tris:,} >= current {n_tris:,}, copying without simplification")
            scene.export(str(output_path))
            return

        logger.info(f"Simplifying to {tgt:,} triangles...")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=tgt,
            preservetopology=True,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=True,
            planarquadric=True,
        )

        ms.save_current_mesh(str(tmp_obj))
        simplified_scene = trimesh.load(str(tmp_obj), force='scene')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simplified_scene.export(str(output_path))
        
        final_tris = ms.current_mesh().face_number()
        reduction = (1 - final_tris / n_tris) * 100
        logger.success(f"Saved to {output_path}")
        logger.info(f"Final: {final_tris:,} triangles ({reduction:.1f}% reduction)")


def main(
    input: Path,
    output: Path,
    target_tris: int,
) -> None:
    """Simplify a single GLB/GLTF file.
    
    Args:
        input: Input GLB/GLTF file
        output: Output GLB/GLTF file
        target_tris: Target number of triangles
    """
    simplify_glb(input, output, target_tris)


if __name__ == "__main__":
    tyro.cli(main)
