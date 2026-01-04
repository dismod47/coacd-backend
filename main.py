"""
CoACD Backend Service
Convex decomposition API using the CoACD library
Following: https://github.com/SarahWeiii/CoACD
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import coacd
import time
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CoACD Decomposition API",
    description="Convex decomposition service using CoACD library",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CoACDParams(BaseModel):
    """Parameters for CoACD decomposition"""
    threshold: float = Field(default=0.05, ge=0.01, le=1.0, description="Concavity threshold (0.01-1.0)")
    max_convex_hull: int = Field(default=-1, description="Max convex hulls (-1 for no limit)")
    preprocess_mode: str = Field(default="auto", description="Preprocessing mode: auto, on, off")
    preprocess_resolution: int = Field(default=50, ge=20, le=100, description="Preprocess resolution (20-100)")
    mcts_iterations: int = Field(default=100, ge=60, le=2000, description="MCTS iterations (60-2000)")
    mcts_depth: int = Field(default=3, ge=2, le=7, description="MCTS depth (2-7)")
    mcts_nodes: int = Field(default=20, ge=10, le=40, description="MCTS nodes (10-40)")
    resolution: int = Field(default=2000, ge=1000, le=10000, description="Sampling resolution (1000-10000)")
    seed: int = Field(default=0, description="Random seed (0 for random)")


class MeshData(BaseModel):
    """Input mesh data"""
    vertices: List[float]  # Flat array [x,y,z,x,y,z,...]
    indices: List[int]     # Triangle indices


class DecomposeRequest(BaseModel):
    """Request body for decomposition"""
    mesh: MeshData
    params: Optional[CoACDParams] = None


class ConvexHullResult(BaseModel):
    """Single convex hull result"""
    id: str
    vertices: List[float]
    indices: List[int]
    volume: float
    centroid: List[float]
    bounding_box: dict


class DecomposeResponse(BaseModel):
    """Response from decomposition"""
    hulls: List[ConvexHullResult]
    total_volume: float
    original_volume: float
    volume_error: float
    compute_time_ms: float
    params: dict


def calculate_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Calculate mesh volume using divergence theorem"""
    volume = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # Signed volume of tetrahedron with origin
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    return abs(volume)


def generate_id() -> str:
    """Generate unique hull ID"""
    import random
    import string
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"hull_{int(time.time())}_{suffix}"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "CoACD Decomposition API"}


@app.get("/health")
async def health():
    """Health check for Render"""
    return {"status": "healthy"}


@app.post("/decompose", response_model=DecomposeResponse)
async def decompose(request: DecomposeRequest):
    """
    Perform convex decomposition on input mesh using CoACD.

    Following CoACD usage:
    ```python
    mesh = coacd.Mesh(vertices, faces)
    parts = coacd.run_coacd(mesh)
    ```
    """
    start_time = time.time()

    try:
        # Parse input mesh
        vertices_flat = np.array(request.mesh.vertices, dtype=np.float64)
        indices_flat = np.array(request.mesh.indices, dtype=np.int32)

        # Reshape to proper format
        num_vertices = len(vertices_flat) // 3
        num_faces = len(indices_flat) // 3

        if num_vertices < 4:
            raise HTTPException(status_code=400, detail="Mesh must have at least 4 vertices")
        if num_faces < 4:
            raise HTTPException(status_code=400, detail="Mesh must have at least 4 faces")

        vertices = vertices_flat.reshape(-1, 3)
        faces = indices_flat.reshape(-1, 3)

        logger.info(f"Processing mesh: {num_vertices} vertices, {num_faces} faces")

        # Calculate original volume
        original_volume = calculate_mesh_volume(vertices, faces)

        # Get parameters
        params = request.params or CoACDParams()

        # Create CoACD mesh
        # Following: mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        mesh = coacd.Mesh(vertices, faces)

        # Run CoACD decomposition
        # Following: parts = coacd.run_coacd(mesh)
        # Parameter names match CoACD Python API (singular forms)
        parts = coacd.run_coacd(
            mesh,
            threshold=params.threshold,
            max_convex_hull=params.max_convex_hull,
            preprocess_mode=params.preprocess_mode,
            prep_resolution=params.preprocess_resolution,
            mcts_iteration=params.mcts_iterations,
            mcts_max_depth=params.mcts_depth,
            mcts_node=params.mcts_nodes,
            resolution=params.resolution,
            seed=params.seed if params.seed != 0 else None
        )

        logger.info(f"CoACD produced {len(parts)} convex hulls")

        # Process results
        hulls = []
        total_volume = 0.0

        for i, part in enumerate(parts):
            part_verts = np.array(part[0], dtype=np.float64)  # vertices
            part_faces = np.array(part[1], dtype=np.int32)    # faces

            # Calculate hull properties
            hull_volume = calculate_mesh_volume(part_verts, part_faces)
            total_volume += hull_volume

            # Calculate centroid
            centroid = part_verts.mean(axis=0).tolist()

            # Calculate bounding box
            bbox_min = part_verts.min(axis=0).tolist()
            bbox_max = part_verts.max(axis=0).tolist()

            # Flatten for JSON response
            hull = ConvexHullResult(
                id=generate_id(),
                vertices=part_verts.flatten().tolist(),
                indices=part_faces.flatten().tolist(),
                volume=hull_volume,
                centroid=centroid,
                bounding_box={"min": bbox_min, "max": bbox_max}
            )
            hulls.append(hull)

        # Calculate volume error
        volume_error = 0.0
        if original_volume > 0:
            volume_error = abs(total_volume - original_volume) / original_volume * 100

        compute_time_ms = (time.time() - start_time) * 1000

        logger.info(f"Decomposition complete in {compute_time_ms:.1f}ms")

        return DecomposeResponse(
            hulls=hulls,
            total_volume=total_volume,
            original_volume=original_volume,
            volume_error=volume_error,
            compute_time_ms=compute_time_ms,
            params=params.model_dump()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decomposition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decomposition failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
