import numpy as np
import cv2
import imageio
from PIL import Image
import argparse

def pad_to_even(frame):
    """Pad frame to even dimensions for video encoding."""
    H, W = frame.shape[:2]
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        pad = ((0, pad_h), (0, pad_w)) if frame.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
        frame = np.pad(frame, pad, mode='edge')
    return frame

def compose_sim_to_real_video(key_name, demo_id):
    """
    Composite simulated video onto real background using depth-based z-buffering.
    Only renders simulated objects that are closer than the real background.
    """
    # ===== CONSTRUCT PATHS FROM CLI ARGUMENTS =====
    base_path = f'/app/outputs/{key_name}'
    demo_path = f'{base_path}/demos/demo_{demo_id}/env_000'
    
    SIM_VIDEO_PATH = f'{demo_path}/sim_video.mp4'
    DEPTH_VIDEO_PATH = f'{demo_path}/depth_video.mp4'
    MASK_VIDEO_PATH = f'{demo_path}/mask_video.mp4'
    REAL_BACKGROUND_PATH = f'{base_path}/reconstruction/background.jpg'
    REAL_DEPTH_PATH = f'{base_path}/reconstruction/image_background_depth.tiff'
    OUTPUT_PATH = f'{demo_path}/real_video.mp4'
    
    USE_DEPTH_COMPARISON = False               # Enable depth-based occlusion (False = mask only)
    DEBUG = True                                # Print debug information
    
    # Depth scaling options (only used if USE_DEPTH_COMPARISON = True)
    NORMALIZE_DEPTHS = True                    # Normalize both depths to 0-1 range
    DEPTH_TOLERANCE = 0.0                      # Tolerance for depth comparison (e.g., 0.1 = 10% tolerance)
    SIM_DEPTH_SCALE = 1.0                      # Scale factor for sim depth (if not normalizing)
    SIM_DEPTH_OFFSET = 0.0                     # Offset for sim depth (if not normalizing)
    # ============================================
    
    # Load real background image and depth
    print(f"Loading real background: {REAL_BACKGROUND_PATH}")
    real_img = np.array(Image.open(REAL_BACKGROUND_PATH))
    
    H, W, _ = real_img.shape
    print(f"Background size: {W}x{H}")
    
    # Load real depth only if depth comparison is enabled
    if USE_DEPTH_COMPARISON:
        print(f"Loading real depth: {REAL_DEPTH_PATH}")
        real_depth = np.array(Image.open(REAL_DEPTH_PATH)).astype(np.float32)
        
        # Normalize real depth if enabled
        if NORMALIZE_DEPTHS:
            # Handle negative or zero depths
            valid_mask = real_depth > 0
            if valid_mask.any():
                real_depth_min = real_depth[valid_mask].min()
                real_depth_max = real_depth.max()
            else:
                real_depth_min = real_depth.min()
                real_depth_max = real_depth.max()
            
            real_depth_normalized = np.zeros_like(real_depth)
            real_depth_normalized = (real_depth - real_depth_min) / (real_depth_max - real_depth_min + 1e-8)
            
            if DEBUG:
                print(f"Real depth original range: [{real_depth.min():.2f}, {real_depth.max():.2f}]")
                print(f"Real depth normalized range: [{real_depth_normalized.min():.4f}, {real_depth_normalized.max():.4f}]")
            real_depth = real_depth_normalized
    else:
        print("Depth comparison disabled - using mask only")
        real_depth = None
    
    # Open videos
    print(f"Loading simulated video: {SIM_VIDEO_PATH}")
    rgb_reader = imageio.get_reader(SIM_VIDEO_PATH)
    print(f"Loading mask video: {MASK_VIDEO_PATH}")
    mask_reader = imageio.get_reader(MASK_VIDEO_PATH)
    
    # Load depth video only if depth comparison is enabled
    if USE_DEPTH_COMPARISON:
        print(f"Loading depth video: {DEPTH_VIDEO_PATH}")
        depth_reader = imageio.get_reader(DEPTH_VIDEO_PATH)
    else:
        depth_reader = None
    
    # Get metadata from original video
    N = rgb_reader.count_frames()
    fps = rgb_reader.get_meta_data()['fps']
    print(f"Processing {N} frames at {fps} FPS...")
    
    composed_images = []
    
    for i in range(N):
        # Read frames
        sim_rgb = rgb_reader.get_data(i)
        sim_mask = mask_reader.get_data(i)
        
        # Read depth frame only if depth comparison is enabled
        if USE_DEPTH_COMPARISON:
            sim_depth = depth_reader.get_data(i)
            
            # Convert depth to grayscale if needed
            if sim_depth.ndim == 3:
                sim_depth = cv2.cvtColor(sim_depth, cv2.COLOR_RGB2GRAY)
            sim_depth = sim_depth.astype(np.float32)
            
            # Normalize or scale sim depth
            if NORMALIZE_DEPTHS:
                sim_depth_min = sim_depth[sim_depth > 0].min() if (sim_depth > 0).any() else 0
                sim_depth_max = sim_depth.max()
                sim_depth = (sim_depth - sim_depth_min) / (sim_depth_max - sim_depth_min + 1e-8)
            else:
                sim_depth = sim_depth * SIM_DEPTH_SCALE + SIM_DEPTH_OFFSET
        else:
            sim_depth = None
        
        # Convert mask to binary (grayscale > 127 = foreground)
        if sim_mask.ndim == 3:
            sim_mask = cv2.cvtColor(sim_mask, cv2.COLOR_RGB2GRAY)
        sim_mask = sim_mask > 127
        
        # Resize sim frames to match real background if needed
        if sim_rgb.shape[:2] != (H, W):
            sim_rgb = cv2.resize(sim_rgb, (W, H))
            sim_mask = cv2.resize(sim_mask.astype(np.uint8), (W, H)) > 0
            if USE_DEPTH_COMPARISON and sim_depth is not None:
                sim_depth = cv2.resize(sim_depth, (W, H))
        
        # Debug info for first frame
        if DEBUG and i == 0:
            print(f"\n=== DEBUG INFO (Frame 0) ===")
            print(f"Sim RGB shape: {sim_rgb.shape}, dtype: {sim_rgb.dtype}")
            if USE_DEPTH_COMPARISON:
                print(f"Sim depth shape: {sim_depth.shape}, range: [{sim_depth.min():.4f}, {sim_depth.max():.4f}]")
                print(f"Real depth shape: {real_depth.shape}, range: [{real_depth.min():.4f}, {real_depth.max():.4f}]")
            print(f"Mask shape: {sim_mask.shape}, True pixels: {sim_mask.sum()}/{sim_mask.size} ({100*sim_mask.sum()/sim_mask.size:.2f}%)")
            if USE_DEPTH_COMPARISON:
                closer = sim_depth < real_depth
                print(f"Depth comparison (sim < real): {closer.sum()} pixels ({100*closer.sum()/closer.size:.2f}%)")
                combined = sim_mask & closer
                print(f"Combined mask (mask AND closer): {combined.sum()} pixels ({100*combined.sum()/combined.size:.2f}%)")
                
                # Show some sample depth values where mask is True
                if sim_mask.sum() > 0:
                    mask_indices = np.where(sim_mask)
                    sample_idx = min(10, len(mask_indices[0]))
                    print(f"\nSample depth values in masked region (first {sample_idx} pixels):")
                    for j in range(sample_idx):
                        y, x = mask_indices[0][j], mask_indices[1][j]
                        print(f"  Pixel ({y},{x}): sim_depth={sim_depth[y,x]:.4f}, real_depth={real_depth[y,x]:.4f}, closer={sim_depth[y,x] < real_depth[y,x]}")
            print("="*30 + "\n")
        
        # Determine which pixels to composite
        if USE_DEPTH_COMPARISON:
            # Only show simulated object where it's closer than real background (with tolerance)
            # AND within the mask region
            closer_mask = sim_mask & (sim_depth < (real_depth + DEPTH_TOLERANCE))
        else:
            # Just use mask without depth comparison
            closer_mask = sim_mask
        
        # Compose: real background + simulated foreground (where closer)
        composed = real_img.copy()
        composed = pad_to_even(composed)
        composed[closer_mask] = sim_rgb[closer_mask]
        
        composed_images.append(composed)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{N} frames")
    
    # Save composed video
    print(f"Saving composed video to: {OUTPUT_PATH}")
    writer = imageio.get_writer(OUTPUT_PATH, fps=fps, macro_block_size=None)
    for frame in composed_images:
        writer.append_data(frame)
    writer.close()
    
    print(f"Done! Saved {len(composed_images)} frames to {OUTPUT_PATH} at {fps} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Composite sim video onto real background')
    parser.add_argument('--key_name', type=str, required=True, help='Key name (e.g., demo_video)')
    parser.add_argument('--demo_id', type=int, required=True, help='Demo ID (e.g., 6)')
    
    args = parser.parse_args()
    compose_sim_to_real_video(args.key_name, args.demo_id)