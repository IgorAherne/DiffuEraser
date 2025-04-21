import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image # Assuming Pillow (PIL fork) is used for Image
from tqdm import tqdm

# Default reasonable color properties if probing fails or is ambiguous
DEFAULT_COLOR_RANGE = 'tv'
DEFAULT_COLOR_SPACE = 'bt709'
DEFAULT_COLOR_PRIMARIES = 'bt709'
DEFAULT_COLOR_TRC = 'bt709'

def write_video_with_ffmpeg(self,
                           frames_list,
                           output_path,
                           fps,
                           size, # Expected as (width, height) tuple
                           original_video_path=None, # Optional: For probing target YUV tags
                           ffmpeg_exe_path='ffmpeg',
                           ffprobe_exe_path='ffprobe',
                           # Renamed quality_mode to be more descriptive
                           save_mode='lossless_ffv1_rgb'): # Options below
    """
    Saves a list of NumPy RGB uint8 frames to a video file using FFmpeg,
    prioritizing lossless quality for intermediate editing.
    Args:
        frames_list: List of NumPy arrays (uint8, RGB, HxWxC).
        output_path: Path to save the output video.
        fps: Frames per second for the output video.
        size: Tuple of (width, height) for the output video.
        original_video_path: Path to the original video to probe for setting
                                output tags when saving to a YUV format. Ignored
                                if save_mode targets RGB output.
        ffmpeg_exe_path: Path or name of the ffmpeg executable.
        ffprobe_exe_path: Path or name of the ffprobe executable.
        save_mode (str): Defines the codec and color format. Options:
            'lossless_ffv1_rgb': (Recommended Highest Fidelity) Saves as RGB using FFV1.
                                    No color space conversion during save.
            'lossless_h264_yuv444p': Saves as YUV444p using lossless H.264.
                                        Performs RGB->YUV conversion via scale filter.
            'lossless_ffv1_yuv444p': Saves as YUV444p using FFV1.
                                     Performs RGB->YUV conversion via scale filter.
            'high_quality_lossy': Saves as YUV420p using high-quality H.264 (CRF 17).
                                    Performs RGB->YUV conversion via scale filter.
    """
    # --- Input Validation and Setup ---
    if not frames_list:
        print("Error: No frames provided to save.")
        return
    if not isinstance(frames_list, (list, tuple)) or not isinstance(frames_list[0], np.ndarray):
            raise TypeError("frames_list must be a list/tuple of NumPy arrays.")
    if frames_list[0].dtype != np.uint8:
            print(f"Warning: Input frames have dtype {frames_list[0].dtype}, expected uint8. Will attempt conversion.")
            # Conversion attempt happens later in the write thread for efficiency

    height, width = frames_list[0].shape[:2]
    if size != (width, height):
            print(f"Warning: Provided size {size} differs from frame dimensions {(width, height)}. Using frame dimensions.")
            size = (width, height) # Use actual frame dimensions

    print(f"Saving video with mode: '{save_mode}'")

    # --- 1. Probe Original Video (Only needed for YUV output modes) ---
    target_range = DEFAULT_COLOR_RANGE
    target_space = DEFAULT_COLOR_SPACE
    target_primaries = DEFAULT_COLOR_PRIMARIES
    target_trc = DEFAULT_COLOR_TRC

    # Probing is only strictly necessary if we're converting to YUV and need to match original tags
    needs_yuv_probe = 'yuv' in save_mode or 'lossy' in save_mode
    if needs_yuv_probe and original_video_path and Path(original_video_path).is_file():
        try:
            print(f"Probing original video for target YUV color tags: {original_video_path}")
            probe_command = [
                ffprobe_exe_path, '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=color_range,color_space,color_primaries,color_trc', # Only need color tags now
                '-of', 'default=noprint_wrappers=1:nokey=0', str(original_video_path)
            ]
            probe_result = subprocess.run(probe_command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=60)
            probed_data = {}
            for line in probe_result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    if value and value != 'N/A': probed_data[key.strip()] = value.strip()
            print(f"Probed target YUV tags: {probed_data}")

            # Update target tags if probed, otherwise keep defaults
            target_range = probed_data.get('color_range', target_range)
            target_space = probed_data.get('color_space', target_space)
            target_primaries = probed_data.get('color_primaries', target_primaries)
            target_trc = probed_data.get('color_trc', target_trc)

            # Basic validation (can be expanded)
            valid_ranges = ['tv', 'pc', 'mpeg', 'jpeg']
            if target_range not in valid_ranges:
                    print(f"Warning: Probed color_range '{target_range}' unrecognized. Using default '{DEFAULT_COLOR_RANGE}'.")
                    target_range = DEFAULT_COLOR_RANGE
            # Add similar validation for space, primaries, trc if desired

        except Exception as e:
            print(f"Warning: Probing original video for YUV tags failed. Using default tags (Range={target_range}, Space={target_space}, etc.). Error: {e}")
    elif needs_yuv_probe:
            print(f"Warning: YUV save mode selected but no valid original video path provided for probing. Using default YUV tags (Range={target_range}, Space={target_space}, etc.).")

    # --- 2. Construct FFmpeg Command based on save_mode ---

    # Input settings (common to all modes)
    command_input = [
        ffmpeg_exe_path, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{size[0]}x{size[1]}',
        '-pix_fmt', 'rgb24', # Input from pipe is uint8 RGB
        '-r', str(fps),
        '-i', '-', # Read from stdin
    ]

    command_filters = []     # Filters like scale
    command_output_tags = [] # Output format/color tags
    command_codec_quality = [] # Codec and quality settings

    # --- Configure based on save_mode ---
    if save_mode == 'lossless_ffv1_rgb':
        print("Configuring for Lossless FFV1 (RGB)...")
        codec = 'ffv1'
        output_pix_fmt = 'rgb24' # Store RGB directly
        # No scale filter needed - direct data passthrough
        command_filters = []
        # Tag output as standard PC range, BT.709 (sRGB standard)
        command_output_tags = [
            '-pix_fmt', output_pix_fmt,
            '-color_range', RGB_OUT_RANGE,
            '-colorspace', RGB_OUT_SPACE,
            '-color_primaries', RGB_OUT_PRIMARIES,
            '-color_trc', RGB_OUT_TRC,
        ]
        command_codec_quality = ['-c:v', codec] # FFV1 is inherently lossless

    elif save_mode == 'lossless_h264_yuv444p':
        print("Configuring for Lossless H.264 (YUV444p)...")
        codec = 'libx264'
        output_pix_fmt = 'yuv444p' # Needs full chroma for lossless from RGB
        # Use scale filter for RGB (PC/709 assumed) -> Target YUV conversion
        vf_string = f"scale=in_range=pc:out_range={target_range}:in_color_matrix=bt709:out_color_matrix={target_space}"
        print(f"Using scale filter: {vf_string}")
        command_filters = ['-vf', vf_string]
        # Tag output with probed/default YUV characteristics
        command_output_tags = [
            '-pix_fmt', output_pix_fmt,
            '-color_range', target_range,
            '-colorspace', target_space,
            '-color_primaries', target_primaries,
            '-color_trc', target_trc,
        ]
        # Use -qp 0 for H.264 lossless, ultrafast preset is fine
        command_codec_quality = ['-c:v', codec, '-preset', 'ultrafast', '-qp', '0']

    elif save_mode == 'lossless_ffv1_yuv444p':
        print("Configuring for Lossless FFV1 (YUV444p)...")
        codec = 'ffv1'
        output_pix_fmt = 'yuv444p' # Store YUV444p losslessly
        # Use scale filter for RGB (PC/709 assumed) -> Target YUV conversion
        vf_string = f"scale=in_range=pc:out_range={target_range}:in_color_matrix=bt709:out_color_matrix={target_space}"
        print(f"Using scale filter: {vf_string}")
        command_filters = ['-vf', vf_string]
        # Tag output with probed/default YUV characteristics
        command_output_tags = [
            '-pix_fmt', output_pix_fmt,
            '-color_range', target_range,
            '-colorspace', target_space,
            '-color_primaries', target_primaries,
            '-color_trc', target_trc,
        ]
        command_codec_quality = ['-c:v', codec] # FFV1 is inherently lossless

    elif save_mode == 'high_quality_lossy':
        print("Configuring for High Quality Lossy H.264 (YUV420p)...")
        codec = 'libx264'
        output_pix_fmt = 'yuv420p' # Standard lossy format
        # Use scale filter for RGB (PC/709 assumed) -> Target YUV conversion
        vf_string = f"scale=in_range=pc:out_range={target_range}:in_color_matrix=bt709:out_color_matrix={target_space}"
        print(f"Using scale filter: {vf_string}")
        command_filters = ['-vf', vf_string]
        # Tag output with probed/default YUV characteristics
        command_output_tags = [
            '-pix_fmt', output_pix_fmt,
            '-color_range', target_range,
            '-colorspace', target_space,
            '-color_primaries', target_primaries,
            '-color_trc', target_trc,
        ]
        # High quality lossy settings
        command_codec_quality = ['-c:v', codec, '-preset', 'slow', '-crf', '17']

    else:
        raise ValueError(f"Unsupported save_mode: '{save_mode}'. Valid modes: "
                            "'lossless_ffv1_rgb', 'lossless_h264_yuv444p', "
                            "'lossless_ffv1_yuv444p', 'high_quality_lossy'")

    # --- Final Command Assembly ---
    command = (command_input + command_filters + command_output_tags +
                command_codec_quality + ['-r', str(fps), str(output_path)])

    print("\n--- Running FFmpeg Save Command ---")
    # Print command safely for debugging
    printable_command = []
    for item in command:
        if os.path.sep in item and Path(item).name in item:
            printable_command.append(repr(item))
        else:
            printable_command.append(item)
    print(" ".join(printable_command))
    print("-----------------------------------\n")

    # --- 3. Setup Threading and Run Process ---
    # (This part remains largely the same as the previous robust version)
    q = queue.Queue(maxsize=10) # Add a maxsize to prevent excessive memory usage
    process = None
    writer_exception = None # To capture exceptions from the writer thread

    def _write_frame(process, q, pbar):
        nonlocal writer_exception
        try:
            while True:
                item = q.get()
                if item is None: break # Sentinel found
                frame = item
                try:
                    # Ensure frame is uint8 before converting to bytes
                    if frame.dtype != np.uint8:
                        # This should be rare if input is correct, but handles it
                        frame = frame.astype(np.uint8)
                    # Directly get bytes - no further conversion needed here
                    frame_bytes = frame.tobytes()
                    process.stdin.write(frame_bytes)
                    pbar.update(1)
                except BrokenPipeError:
                        print("\nError: FFmpeg pipe broke (likely process terminated). Stopping writer thread.")
                        # Can't write anymore
                        break
                except Exception as e:
                    print(f"\nError writing frame to FFmpeg pipe: {e}")
                    # Raise or store the exception
                    writer_exception = e
                    break # Stop writing on error
        except Exception as e:
                # Catch errors in the queue logic itself
                print(f"Error in writer thread queue handling: {e}")
                writer_exception = e
        finally:
                print("Writer thread finished.")

    # --- Start FFmpeg Process and Writer Thread ---
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pbar_desc = f"Encoding ({save_mode})"
        pbar = tqdm(total=len(frames_list), desc=pbar_desc, unit="frame")
        writer_thread = threading.Thread(target=_write_frame, args=(process, q, pbar), name="ffmpeg_writer")
        writer_thread.daemon = True
        writer_thread.start()

        # Feed frames to the queue
        for i, frame in enumerate(frames_list):
            if not writer_thread.is_alive():
                print(f"Error: Writer thread died prematurely (frame {i}). Stopping.")
                if writer_exception: # Check if writer stored an exception
                        raise writer_exception from None # Re-raise exception from writer
                else:
                        raise RuntimeError("FFmpeg writer thread terminated unexpectedly.")
            q.put(frame) # Blocks if queue is full (maxsize)

        q.put(None) # Signal end of frames

        # Wait for writer thread to finish processing the queue
        writer_thread.join(timeout=300) # 5 min timeout
        if writer_thread.is_alive():
                print("Warning: Writer thread did not finish within timeout!")
                # Aggressively terminate ffmpeg if writer hangs
                if process.poll() is None: process.terminate()

        pbar.close()

        # Check for exceptions raised in the writer thread
        if writer_exception:
                print("Error occurred in writer thread, re-raising...")
                raise writer_exception from None

        # Close stdin carefully
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                print("Info: Pipe already broken when closing stdin (FFmpeg likely exited).")
            except Exception as e_close:
                print(f"Error closing stdin pipe: {e_close}")

        # Wait for FFmpeg process to finish and get output
        try:
                stdout, stderr = process.communicate(timeout=600) # 10 min timeout for potentially slow lossless encodes
                return_code = process.returncode
        except subprocess.TimeoutExpired:
                print("\nError: FFmpeg process timed out after closing input.")
                if process.poll() is None: process.terminate() # Attempt termination
                raise TimeoutError(f"FFmpeg save process timed out.") from None

        # --- Check final return code and report ---
        if return_code != 0:
            stderr_str = stderr.decode('utf-8', errors='replace').strip()
            print(f"\n--- FFmpeg Save Error (Return Code: {return_code}) ---")
            if stderr_str: print(f"Stderr:\n{stderr_str}")
            else: print("Stderr: <Empty>")
            print("------------------------------------------------------\n")
            raise RuntimeError(f"FFmpeg save process failed with return code {return_code}")
        else:
            print(f"Video successfully saved to: {output_path}")
            # Optional: Log non-empty stdout on success for info
            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            if stdout_str: print(f"\nFFmpeg stdout:\n{stdout_str}")

    except FileNotFoundError:
            # Raised by subprocess.Popen if executable not found
            print(f"Error: ffmpeg executable not found at '{ffmpeg_exe_path}'. Cannot save video.")
            raise
    except Exception as e:
        print(f"\nAn error occurred during video saving: {type(e).__name__}: {e}")
        # Ensure process is terminated if it's still running on error
        if process and process.poll() is None:
                print("Terminating FFmpeg process due to error...")
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill()
                print("FFmpeg process terminated.")
        # Re-raise the exception unless it's one we specifically handle/raise
        if not isinstance(e, (RuntimeError, TimeoutError, FileNotFoundError, BrokenPipeError)):
                raise
    finally:
        # Final cleanup check
        if process and process.poll() is None:
                print("Warning: FFmpeg process may still be running after function exit.")
# --- End of save_video_with_ffmpeg method ---



def read_frames_high_fidelity_ffmpeg(
    video_path,
    max_length=99999.0,
    ffmpeg_exe_path='ffmpeg',
    ffprobe_exe_path='ffprobe'):
    """
    Reads video frames using FFmpeg subprocess pipe, aiming for high fidelity
    by using a 16-bit RGB intermediate and explicit color tagging.
    Relies on subprocess to find executables via path or system PATH.
    Args:
        video_path: Path to the input video file.
        max_length: Maximum duration to read in seconds. Defaults to very large.
        ffmpeg_exe_path: Path or name of the ffmpeg executable.
        ffprobe_exe_path: Path or name of the ffprobe executable.
    Returns:
        tuple: (frames_pil, fps, size, color_info, n_frames_read)
               frames_pil: List of PIL Image objects (RGB, uint8).
               fps: Video frame rate (float).
               size: Tuple of (width, height).
               color_info: Dictionary containing probed/used color properties.
               n_frames_read: Number of frames actually read.
    Raises:
        FileNotFoundError: If ffmpeg/ffprobe is not found by subprocess, or input video missing.
        IOError: If ffprobe or ffmpeg subprocess fails, or pipe errors occur.
        ValueError: If video metadata is invalid or missing crucial info.
    """
    # --- Input Validation and Setup ---
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # --- Executable checks removed ---
    # The script now relies on subprocess.run/Popen finding the executables.
    # If not found, FileNotFoundError will be raised during the subprocess call.

    print(f"Reading video for high fidelity using FFmpeg pipe: {video_path}")

    # --- 1. Probe Video Thoroughly ---
    print("Probing video details with ffprobe...")
    probe_command = [
        ffprobe_exe_path, # Directly use the provided path/name
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,pix_fmt,color_range,color_space,color_primaries,color_trc',
        '-of', 'default=noprint_wrappers=1:nokey=0',
        str(video_path)
    ]
    color_info = { # Initialize with defaults
        'range': DEFAULT_COLOR_RANGE,
        'space': DEFAULT_COLOR_SPACE,
        'primaries': DEFAULT_COLOR_PRIMARIES,
        'trc': DEFAULT_COLOR_TRC,
        'pix_fmt': 'yuv420p' # Assume common default if not probed
    }
    probed_data = {}
    try:
        # FileNotFoundError will be raised here if ffprobe_exe_path is invalid
        probe_result = subprocess.run(probe_command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=60)
        #  parsing and validation logic
        for line in probe_result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                if value and value != 'N/A':
                     probed_data[key.strip()] = value.strip()

        if 'width' not in probed_data or 'height' not in probed_data:
            raise ValueError("Could not probe video width/height.")
        width = int(probed_data['width'])
        height = int(probed_data['height'])
        size = (width, height)

        if 'r_frame_rate' not in probed_data:
             print("Warning: Could not probe r_frame_rate. Defaulting FPS to 30.0.")
             fps = 30.0
        else:
             try:
                 num, den = map(float, probed_data['r_frame_rate'].split('/'))
                 fps = num / den if den != 0 else 30.0
             except Exception as fps_err:
                 print(f"Warning: Could not parse r_frame_rate '{probed_data['r_frame_rate']}'. Defaulting FPS to 30.0. Error: {fps_err}")
                 fps = 30.0

        duration_str = probed_data.get('duration', None)
        duration = float(duration_str) if duration_str else 0

        color_info['pix_fmt'] = probed_data.get('pix_fmt', color_info['pix_fmt'])
        color_info['range'] = probed_data.get('color_range', color_info['range'])
        color_info['space'] = probed_data.get('color_space', color_info['space'])
        color_info['primaries'] = probed_data.get('color_primaries', color_info['primaries'])
        color_info['trc'] = probed_data.get('color_trc', color_info['trc'])

        if color_info['pix_fmt'].startswith('yuvj'):
             print("Interpreting 'yuvj' pix_fmt as standard YUV.")
             color_info['pix_fmt'] = color_info['pix_fmt'].replace('j', '')
             if color_info['range'] != 'tv':
                  print("Assuming 'pc' range for YUVJ format (unless range was explicitly 'tv').")
                  color_info['range'] = 'pc'

        valid_ranges = ['tv', 'pc', 'mpeg', 'jpeg']
        if color_info['range'] not in valid_ranges:
             print(f"Warning: Probed color_range '{color_info['range']}' not in typical list {valid_ranges}. Using default '{DEFAULT_COLOR_RANGE}'.")
             color_info['range'] = DEFAULT_COLOR_RANGE

        print(f"Probed info: Size={size}, FPS={fps:.3f}, Duration={duration:.2f}s")
        print(f"Using Color Info: {color_info}")

    except FileNotFoundError as e:
         # Catch specifically if subprocess couldn't find ffprobe
         raise FileNotFoundError(f"ffprobe executable not found at '{ffprobe_exe_path}' or in PATH.") from e
    except subprocess.CalledProcessError as e:
        raise IOError(f"ffprobe command failed for '{video_path}'. Error: {e.stderr}") from e
    except Exception as e:
        raise IOError(f"Failed during ffprobe processing for '{video_path}'. Error: {e}") from e

    # --- 2. Construct FFmpeg Read Command ---
    # using rgb48le ...
    output_pix_fmt = 'rgb48le'
    bytes_per_channel = 2
    bytes_per_pixel = 3 * bytes_per_channel
    bytes_per_frame = width * height * bytes_per_pixel

    read_command = [
        ffmpeg_exe_path, # Directly use the provided path/name
        # Explicitly declare INPUT characteristics based on probing
        '-color_range', color_info['range'],
        '-colorspace', color_info['space'],
        '-color_primaries', color_info['primaries'],
        '-color_trc', color_info['trc'],
        # Input file
        '-i', str(video_path),
        # Time limit if applicable
        *((['-t', str(max_length)] if 0 < max_length < duration else [])),
        # Output options
        '-f', 'rawvideo',
        '-pix_fmt', output_pix_fmt,
        '-an', '-sn',
        '-v', 'warning',
        '-'
    ]
    print("Running FFmpeg read command: " + " ".join(read_command))

    # --- 3. Execute FFmpeg and Read Frames ---
    process = None
    frames_pil = []
    try:
        # FileNotFoundError will be raised here if ffmpeg_exe_path is invalid
        process = subprocess.Popen(read_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_count = 0
        while True:
            in_bytes = process.stdout.read(bytes_per_frame)
            if not in_bytes: break
            if len(in_bytes) != bytes_per_frame:
                print(f"Warning: Incomplete frame {frame_count} received ({len(in_bytes)}/{bytes_per_frame} bytes). Stopping.")
                break

            frame_np_uint16 = np.frombuffer(in_bytes, dtype=np.uint16).reshape([height, width, 3])
            frame_np_uint8 = (frame_np_uint16 >> 8).astype(np.uint8)
            frames_pil.append(Image.fromarray(frame_np_uint8))
            frame_count += 1

        stdout, stderr = process.communicate(timeout=120)
        return_code = process.returncode

        if return_code != 0:
            stderr_str = stderr.decode('utf-8', errors='replace').strip()
            ignore_errors = ["pipe:", "End of file", "Conversion failed!"]
            is_ignorable = any(msg in stderr_str for msg in ignore_errors)

            if is_ignorable and frame_count > 0:
                 print(f"FFmpeg finished with code {return_code} (Possibly due to -t or pipe close). Stderr contained ignorable message(s).")
            else:
                 print(f"\n--- FFmpeg Read Error (Return Code: {return_code}) ---")
                 print(f"Stderr:\n{stderr_str}")
                 print("------------------------------------------------------\n")
                 raise IOError(f"FFmpeg reading process failed unexpectedly with return code {return_code}")

    except FileNotFoundError as e:
        # Catch specifically if subprocess couldn't find ffmpeg
        raise FileNotFoundError(f"ffmpeg executable not found at '{ffmpeg_exe_path}' or in PATH.") from e
    except Exception as e:
        if process and process.poll() is None:
            print("Terminating FFmpeg process due to error...")
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill()
        raise IOError(f"Error executing FFmpeg or reading pipe: {type(e).__name__} - {e}") from e
    finally:
        if process:
            if process.stdout: process.stdout.close()
            if process.stderr: process.stderr.close()
            if process.poll() is None:
                print("Force killing lingering FFmpeg process...")
                process.kill()

    n_frames_read = len(frames_pil)
    if n_frames_read == 0 and max_length > 0:
         print("Warning: FFmpeg pipe read resulted in 0 frames.")
    else:
         print(f"Successfully read {n_frames_read} frames via FFmpeg pipe (using {output_pix_fmt} intermediate).")

    return frames_pil, fps, size, color_info, n_frames_read