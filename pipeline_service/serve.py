from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import base64

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import pyspz

from config import settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse, TrellisRequest, TrellisParams
from modules import GenerationPipeline
from modules.utils import secure_randint, set_random_seed
from PIL import Image
import io

pipeline = GenerationPipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.startup()
    # warm up pipeline
    temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buffer = io.BytesIO()
    temp_image.save(buffer, format="PNG")
    temp_imge_bytes = buffer.getvalue()
    await run_champion_generation(temp_imge_bytes, -1)
    pipeline._clean_gpu_memory()
    try:
        yield
    finally:
        await pipeline.shutdown()


app = FastAPI(
    title=settings.api_title,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_champion_generation(
    image_bytes: bytes, seed: int
) -> tuple[bytes, int]:
    """
    BATCHED MULTIDIFFUSION WITH AUTOMATIC VARIETY:
    1. Batch generate N candidates with multidiffusion mode
    2. Select the first generated candidate
    
    Important: Even with SAME seed, batching produces DIFFERENT models because:
    - Random noise is generated per sample in the batch
    - Each sample gets sequential values from the RNG stream
    - Result: Natural variety without sacrificing speed!
    """
    import time
    
    if seed < 0:
        seed = secure_randint(0, 10000)
    set_random_seed(seed)

    logger.info("⚡ Preprocessing images (One-time execution)...")
    # START TIMER: Begin when Qwen editing starts
    generation_start = time.time()
    
    (
        left_image_without_background,
        right_image_without_background,
        original_image_without_background,
    ) = await pipeline.prepare_input_images(image_bytes, seed)

    from schemas import TrellisRequest, TrellisParams
    params = TrellisParams.from_settings(settings)

    # STAGE 1: Generate max_candidates shape candidates
    logger.info(f"🎲 STAGE 1: Generating {settings.max_candidates} shape candidates...")
    trellis_req = TrellisRequest(
        images=[left_image_without_background, right_image_without_background, original_image_without_background],
        seed=seed,
        params=params,
    )
    
    shape_candidates = pipeline.trellis.generate_shapes_only(trellis_req, num_samples=settings.max_candidates)
    
    # Use first shape to determine number of texture candidates needed
    reference_voxel_count = shape_candidates[0][1]
    
    # Determine number of texture candidates based on voxel ranges
    num_texture_candidates = settings.candidate_ranges[-1][1]  # Default to last range's value
    for max_voxels, num_candidates in settings.candidate_ranges:
        if reference_voxel_count <= max_voxels:
            num_texture_candidates = num_candidates
            logger.info(f"📊 Voxel count {reference_voxel_count} ≤ {max_voxels} → {num_candidates} candidates")
            break
    else:
        logger.info(f"📊 Voxel count {reference_voxel_count} > {settings.candidate_ranges[-1][0]} → {num_texture_candidates} candidate(s)")
    
    # Select N smallest shapes from the 5 generated
    # Keep track of original indices during sorting
    indexed_shapes = [(i, coords, voxel_count) for i, (coords, voxel_count) in enumerate(shape_candidates)]
    sorted_indexed_shapes = sorted(indexed_shapes, key=lambda x: x[2])  # Sort by voxel count
    selected_indexed_shapes = sorted_indexed_shapes[:num_texture_candidates]  # Take N smallest
    
    logger.success(f"📐 Selected {num_texture_candidates} smallest shapes:")
    for original_idx, coords, voxel_count in selected_indexed_shapes:
        logger.info(f"   Shape {original_idx+1}: {voxel_count} voxels")
    
    # Extract just coords and voxel_count for texture generation
    selected_shapes = [(coords, voxel_count) for _, coords, voxel_count in selected_indexed_shapes]
    
    logger.info(f"🎨 STAGE 2: Generating {num_texture_candidates} textures (one per selected shape)")
    
    # STAGE 2: Generate textures for the selected smallest shapes
    candidates = pipeline.trellis.generate_textures_for_shapes(
        trellis_req,
        selected_shapes
    )
    
    generation_time = time.time() - generation_start
    logger.info(
        f"✅ Generated {len(candidates)} candidate(s) in {generation_time:.2f}s. Returning the first candidate."
    )
    return candidates[0].ply_file, seed


# ---------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/generate_from_base64", response_model=GenerateResponse)
async def generate_from_base64(request: GenerateRequest) -> GenerateResponse:
    """
    Endpoint JSON này giữ nguyên, thường dùng cho test single request.
    """
    try:
        # Lưu ý: generate_gs trong pipeline nên được cập nhật để dùng prepare_input_images bên trong
        result = await pipeline.generate_gs(request)

        compressed_ply_bytes = None
        if result.ply_file_base64 and settings.compression:
            compressed_ply_bytes = pyspz.compress(result.ply_file_base64, workers=1)
            logger.info(f"Compressed PLY size: {len(compressed_ply_bytes)} bytes")

        result.ply_file_base64 = base64.b64encode(
            result.ply_file_base64 if not compressed_ply_bytes else compressed_ply_bytes
        ).decode("utf-8")

        return result
    except Exception as exc:
        logger.exception(f"Error generating task: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
async def generate(
    prompt_image_file: UploadFile = File(...), seed: int = Form(-1)
) -> StreamingResponse:
    """
    Generate PLY (Streaming). Có chạy giải đấu nếu seed = -1.
    """
    try:
        logger.info(
            f"Task received (/generate). Uploading image: {prompt_image_file.filename}"
        )
        image_bytes = await prompt_image_file.read()

        # Gọi hàm chung
        final_ply_bytes, selected_seed = await run_champion_generation(
            image_bytes, seed
        )

        ply_buffer = BytesIO(final_ply_bytes)
        buffer_size = len(ply_buffer.getvalue())
        ply_buffer.seek(0)

        headers = {
            "Content-Length": str(buffer_size),
            "X-Generated-Seed": str(selected_seed),
        }

        async def generate_chunks():
            chunk_size = 1024 * 1024
            while chunk := ply_buffer.read(chunk_size):
                yield chunk

        return StreamingResponse(
            generate_chunks(), media_type="application/octet-stream", headers=headers
        )

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate-spz")
async def generate_spz(
    prompt_image_file: UploadFile = File(...), seed: int = Form(-1)
) -> StreamingResponse:
    """
    Generate SPZ (Compressed). BẮT BUỘC CHẠY GIẢI ĐẤU nếu seed = -1 để lấy model tốt nhất.
    """
    try:
        logger.info(
            f"Task received (/generate-spz). Uploading image: {prompt_image_file.filename}"
        )
        image_bytes = await prompt_image_file.read()

        # 1. Gọi hàm chung để lấy model VÔ ĐỊCH (Champion)
        final_ply_bytes, selected_seed = await run_champion_generation(
            image_bytes, seed
        )

        # 2. Nén thành SPZ
        if final_ply_bytes:
            logger.info("Compressing Champion model to SPZ...")
            ply_compressed_bytes = pyspz.compress(final_ply_bytes, workers=1)
            logger.info(f"Task completed. SPZ size: {len(ply_compressed_bytes)} bytes")

            # (Tùy chọn) Có thể trả về Header thông tin người thắng cuộc
            headers = {
                "X-Generated-Seed": str(selected_seed),
            }

            return StreamingResponse(
                BytesIO(ply_compressed_bytes),
                media_type="application/octet-stream",
                headers=headers,
            )
        else:
            raise HTTPException(status_code=500, detail="Generated content is empty")

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/setup/info")
async def get_setup_info() -> dict:
    try:
        return settings.dict()
    except Exception as e:
        logger.error(f"Failed to get setup info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host=settings.host, port=settings.port, reload=False)
