import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
import gc

from propainter.model.modules.flow_comp_raft import RAFT_bi
from propainter.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from propainter.model.propainter import InpaintGenerator
from propainter.utils.download_util import load_file_from_url
from propainter.utils.video_read_write import read_frames_high_fidelity_ffmpeg, write_video_with_ffmpeg
from propainter.core.utils import to_tensors
from propainter.model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
MaxSideThresh = 960


def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        # Ensure dimensions are at least 8x8 after rounding down
        process_width = max(8, out_size[0] - out_size[0] % 8)
        process_height = max(8, out_size[1] - out_size[1] % 8)
        process_size = (process_width, process_height)
        # Use LANCZOS for higher quality resizing
        frames = [f.resize(process_size, Image.Resampling.LANCZOS) for f in frames] # Use Image.LANCZOS for older PIL
    else:
        # handle case where size is None, ensure divisibility by 8 here too if needed
        out_size = frames[0].size
        process_width = max(8, out_size[0] - out_size[0] % 8)
        process_height = max(8, out_size[1] - out_size[1] % 8)
        process_size = (process_width, process_height)
        if not out_size == process_size:
             frames = [f.resize(process_size, Image.Resampling.LANCZOS) for f in frames] # Use Image.LANCZOS for older PIL
    return frames, process_size, out_size

def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
# read frame-wise masks
def read_mask(mpath, frames_len, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        masks_img = [Image.open(mpath)]
    elif mpath.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        cap = cv2.VideoCapture(mpath)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            if(idx >= frames_len):
                break
            masks_img.append(Image.fromarray(frame))
            idx += 1
        cap.release()
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
            # print(mp)
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * frames_len
        masks_dilated = masks_dilated * frames_len

    return flow_masks, masks_dilated

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class Propainter:
    def __init__(self, propainter_model_dir, device,
                 ffmpeg_path, ffprobe_path):
        self.device = device
        self.ffmpeg_exe_path = ffmpeg_path
        self.ffprobe_exe_path = ffprobe_path
        ##############################################
        # set up RAFT and flow competition model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                        model_dir=propainter_model_dir, progress=True, file_name=None)
        self.fix_raft = RAFT_bi(ckpt_path, device)
        
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                        model_dir=propainter_model_dir, progress=True, file_name=None)
        self.fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(device)
        self.fix_flow_complete.eval()

        ##############################################
        # set up ProPainter model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                        model_dir=propainter_model_dir, progress=True, file_name=None)
        self.model = InpaintGenerator(model_path=ckpt_path).to(device)
        self.model.eval()



    def forward(self, video, mask, output_path,
                save_mode='lossless_ffv1_rgb', # Default to high-fidelity RGB
                resize_ratio=0.6, video_length=None, height=-1, width=-1,
                mask_dilation=4, ref_stride=10, neighbor_length=10, subvideo_length=80,
                raft_iter=20, save_fps=None, # Start save_fps as None
                fp16=True):
        # --- Keep initial setup ---
        use_half = True if fp16 else False
        if self.device == torch.device('cpu'):
            use_half = False

        ################ read input video ################
        print("--- Reading Video using Custom FFmpeg Function ---")
        try:
            # Call the new reader function
            frames_pil, fps_read, size_read, color_info, nframes = read_frames_high_fidelity_ffmpeg(
                video_path=video,
                max_length=video_length if video_length is not None else 99999.0,
                ffmpeg_exe_path=self.ffmpeg_exe_path,
                ffprobe_exe_path=self.ffprobe_exe_path
            )
            print("Read Color Info:", color_info)
            if nframes == 0: # Check if reading actually produced frames
                 raise ValueError("Video reading resulted in 0 frames.")
        except (FileNotFoundError, IOError, ValueError) as e:
            print(f"\n\nFATAL ERROR during video reading: {e}\n")
            return None # Indicate failure
        except Exception as e:
             print(f"\n\nUnexpected FATAL ERROR during video reading: {e}\n")
             return None

        # --- Use the read values ---
        frames = frames_pil # Assign the list of PIL images to 'frames'
        # Determine FPS: use save_fps if provided, otherwise use probed fps
        final_fps = save_fps if save_fps is not None else fps_read
        if final_fps is None or final_fps <= 0: # Added check for invalid probed fps
            print(f"Warning: Invalid FPS determined ({final_fps}). Defaulting to 30.")
            final_fps = 30.0
        # Determine initial size: use user override if provided, otherwise use probed size
        if width != -1 and height != -1:
            size = (width, height)
        else:
            size = size_read # Use the size determined by the reader

        # --- Keep Resizing Logic ---
        longer_edge = max(size[0], size[1])
        if longer_edge > MaxSideThresh:
            scale = MaxSideThresh / longer_edge
            resize_ratio = resize_ratio * scale
        if resize_ratio != 1.0: # Apply resizing only if needed
            print(f"Resizing frames with ratio {resize_ratio:.2f}...")
            target_size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))
            # The resize_frames function correctly returns PIL images
            frames, process_size, out_size = resize_frames(frames, target_size)
            print(f"Resized to process size: {process_size}")
        else:
            # If no resize needed, ensure process_size and out_size are set
            frames, process_size, out_size = resize_frames(frames, None) # Let it calc divisible-by-8 size

        w, h = process_size # Use the actual processing dimensions


        ################ read mask ################ 
        frames_len = len(frames) # Length after potential resizing
        # Pass the actual processing size (w, h) to read_mask
        flow_masks, masks_dilated = read_mask( mask, frames_len, process_size,
                                               flow_mask_dilates=mask_dilation,
                                               mask_dilates=mask_dilation )

        ################ adjust input ################ 
        # Keep ori_frames_inp based on the *current* list of PIL frames (potentially resized)
        ori_frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        # Keep tensor conversion logic
        frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
        flow_masks = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated  =  frames.to(self.device), flow_masks.to(self.device), masks_dilated.to(self.device)
 
        ##############################################
        # ProPainter inference
        ##############################################
        video_length = frames.size(1)
        print(f'Priori generating: [{video_length} frames]...')
        with torch.no_grad():
            # ---- compute flow ----
            new_longer_edge = max(frames.size(-1), frames.size(-2))
            if new_longer_edge <= 640: 
                short_clip_len = 12
            elif new_longer_edge <= 720: 
                short_clip_len = 8
            elif new_longer_edge <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            # use fp32 for RAFT
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:,f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                    
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                    
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=raft_iter)
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()

            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()
          
            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + subvideo_length)
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        flow_masks[:, s_f:e_f+1])
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        pred_flows_bi_sub, 
                        flow_masks[:, s_f:e_f+1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()
                    
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()
                

            masks_dilated_ori = masks_dilated.clone()
            # ---- Pre-propagation ----
            subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
            if(len(frames[0]))>subvideo_length_img_prop: # perform propagation only when length of frames is larger than subvideo_length_img_prop
                sample_rate = len(frames[0])//(subvideo_length_img_prop//2)
                index_sample =  list(range(0, len(frames[0]), sample_rate))
                sample_frames =  torch.stack([frames[0][i].to(torch.float32) for i in index_sample]).unsqueeze(0) # use fp32 for RAFT
                sample_masks_dilated = torch.stack([masks_dilated[0][i] for i in index_sample]).unsqueeze(0)
                sample_flow_masks =  torch.stack([flow_masks[0][i] for i in index_sample]).unsqueeze(0)
  
                ## recompute flow for sampled frames
                # use fp32 for RAFT
                sample_video_length = sample_frames.size(1)
                if sample_frames.size(1) > short_clip_len:
                    gt_flows_f_list, gt_flows_b_list = [], []
                    for f in range(0, sample_video_length, short_clip_len):
                        end_f = min(sample_video_length, f + short_clip_len)
                        if f == 0:
                            flows_f, flows_b = self.fix_raft(sample_frames[:,f:end_f], iters=raft_iter)
                        else:
                            flows_f, flows_b = self.fix_raft(sample_frames[:,f-1:end_f], iters=raft_iter)
                        
                        gt_flows_f_list.append(flows_f)
                        gt_flows_b_list.append(flows_b)
                        torch.cuda.empty_cache()
                        
                    gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                    gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                    sample_gt_flows_bi = (gt_flows_f, gt_flows_b)
                else:
                    sample_gt_flows_bi = self.fix_raft(sample_frames, iters=raft_iter)
                    torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()

                if use_half:
                    sample_frames, sample_flow_masks, sample_masks_dilated = sample_frames.half(), sample_flow_masks.half(), sample_masks_dilated.half()
                    sample_gt_flows_bi = (sample_gt_flows_bi[0].half(), sample_gt_flows_bi[1].half())

                # ---- complete flow ----
                flow_length = sample_gt_flows_bi[0].size(1)
                if flow_length > subvideo_length:
                    pred_flows_f, pred_flows_b = [], []
                    pad_len = 5
                    for f in range(0, flow_length, subvideo_length):
                        s_f = max(0, f - pad_len)
                        e_f = min(flow_length, f + subvideo_length + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(flow_length, f + subvideo_length)
                        pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                            (sample_gt_flows_bi[0][:, s_f:e_f], sample_gt_flows_bi[1][:, s_f:e_f]), 
                            sample_flow_masks[:, s_f:e_f+1])
                        pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                            (sample_gt_flows_bi[0][:, s_f:e_f], sample_gt_flows_bi[1][:, s_f:e_f]), 
                            pred_flows_bi_sub, 
                            sample_flow_masks[:, s_f:e_f+1])

                        pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                        pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                        torch.cuda.empty_cache()
                        
                    pred_flows_f = torch.cat(pred_flows_f, dim=1)
                    pred_flows_b = torch.cat(pred_flows_b, dim=1)
                    sample_pred_flows_bi = (pred_flows_f, pred_flows_b)
                else:
                    sample_pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(sample_gt_flows_bi, sample_flow_masks)
                    sample_pred_flows_bi = self.fix_flow_complete.combine_flow(sample_gt_flows_bi, sample_pred_flows_bi, sample_flow_masks)
                    torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                
                masked_frames = sample_frames * (1 - sample_masks_dilated)
                
                if sample_video_length > subvideo_length_img_prop:
                    updated_frames, updated_masks = [], []
                    pad_len = 10
                    for f in range(0, sample_video_length, subvideo_length_img_prop):
                        s_f = max(0, f - pad_len)
                        e_f = min(sample_video_length, f + subvideo_length_img_prop + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(sample_video_length, f + subvideo_length_img_prop)

                        b, t, _, _, _ = sample_masks_dilated[:, s_f:e_f].size()
                        pred_flows_bi_sub = (sample_pred_flows_bi[0][:, s_f:e_f-1], sample_pred_flows_bi[1][:, s_f:e_f-1])
                        prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                            pred_flows_bi_sub, 
                                                                            sample_masks_dilated[:, s_f:e_f], 
                                                                            'nearest')
                        updated_frames_sub = sample_frames[:, s_f:e_f] * (1 - sample_masks_dilated[:, s_f:e_f]) + \
                                            prop_imgs_sub.view(b, t, 3, h, w) * sample_masks_dilated[:, s_f:e_f]
                        updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                        
                        updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                        updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                        torch.cuda.empty_cache()
                        
                    updated_frames = torch.cat(updated_frames, dim=1)
                    updated_masks = torch.cat(updated_masks, dim=1)
                else:
                    b, t, _, _, _ = sample_masks_dilated.size()
                    prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, sample_pred_flows_bi, sample_masks_dilated, 'nearest')
                    updated_frames = sample_frames * (1 - sample_masks_dilated) + prop_imgs.view(b, t, 3, h, w) * sample_masks_dilated
                    updated_masks = updated_local_masks.view(b, t, 1, h, w)
                    torch.cuda.empty_cache()

                ## replace input frames/masks with updated frames/masks 
                for i,index in enumerate(index_sample):
                    frames[0][index] = updated_frames[0][i]
                    masks_dilated[0][index] = updated_masks[0][i]


            # ---- frame-by-frame image propagation ----
            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                        pred_flows_bi_sub, 
                                                                        masks_dilated[:, s_f:e_f], 
                                                                        'nearest')
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()            
                
        comp_frames = [None] * video_length

        neighbor_stride = neighbor_length // 2
        if video_length > subvideo_length:
            ref_num = subvideo_length // ref_stride
        else:
            ref_num = -1
        
        torch.cuda.empty_cache()
        # ---- feature propagation + transformer ----
        for f in tqdm(range(0, video_length, neighbor_stride)):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)

                ## compose with input frames
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated_ori[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)  # use original mask
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames_inp[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else: 
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
            
            torch.cuda.empty_cache()

        ##############################################
        # Saving Video
        ##############################################
        print(f"--- Saving Video using Custom FFmpeg Function (Mode: {save_mode}) ---")
        if comp_frames and comp_frames[0] is not None:
            # Final output size is the processing size (w, h)
            final_output_size = (w, h)
            final_frames_to_save = []
            print("Ensuring final frames are uint8 RGB...")
            # Use comp_frames from the inference loop
            for frame_data in tqdm(comp_frames, desc="Preparing Save"):
                if frame_data.dtype != np.uint8:
                     # Clip and convert if inference outputted floats
                     frame_uint8 = np.clip(frame_data, 0, 255).astype(np.uint8)
                else:
                     frame_uint8 = frame_data
                # Ensure it's RGB (it should be from the inference code)
                if frame_uint8.shape[-1] != 3:
                     print(f"Warning: Frame data has unexpected shape {frame_uint8.shape}. Skipping.")
                     continue
                final_frames_to_save.append(frame_uint8) # Should be HxWxC RGB uint8

            if not final_frames_to_save:
                 print("Error: No valid frames available after final preparation.")
                 output_path = None
            else:
                 try:
                     # Call the new write function
                     write_video_with_ffmpeg(
                         frames_list=final_frames_to_save,
                         output_path=output_path,
                         fps=final_fps, # Use the calculated final FPS
                         size=final_output_size, # Use the processing size (w, h)
                         original_video_path=video, # Pass original video for YUV tag probing
                         ffmpeg_exe_path=self.ffmpeg_exe_path, # Use stored path
                         ffprobe_exe_path=self.ffprobe_exe_path, # Use stored path
                         save_mode=save_mode # Pass the mode selected by the user/default
                     )
                     print(f"Output video saved to {output_path}")
                 except Exception as e:
                     print(f"ERROR during custom video saving: {e}")
                     output_path = None # Indicate failure
        else:
             print("Error: No completed frames generated by inference.")
             output_path = None

        torch.cuda.empty_cache()
        gc.collect() # Add extra garbage collect just in case

        return output_path # Return the path if successful, None otherwise



if __name__ == '__main__':
    # Define paths (or get from argparse if you make this a runnable script)
    ffmpeg_path  = 'C:/_myDrive/repos/auto-vlog/AutoVlogProj/bin/ffmpeg.exe' # Example path
    ffprobe_path = 'C:/_myDrive/repos/auto-vlog/AutoVlogProj/bin/ffprobe.exe'# Example path

    propainter_model_dir = "weights/propainter"
    video_path = "examples/example1/video.mp4"
    mask_path =  "examples/example1/mask.mp4"

    output_path = "results/inpainted.mp4" # Example output path

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = get_device()

    # *** Pass the paths during instantiation ***
    propainter = Propainter(propainter_model_dir,
                            device=device,
                            ffmpeg_path=ffmpeg_path,  # Pass the path
                            ffprobe_path=ffprobe_path) # Pass the path

    print(f"Running ProPainter on {video_path}...")
    # *** Specify the desired save_mode in the forward call ***
    res = propainter.forward(video_path,
                             mask_path,
                             output_path,
                             save_mode='lossless_ffv1_rgb') # Or another mode
    if res:
        print(f"Processing complete. Output saved to: {res}")
    else:
        print("Processing failed.")
