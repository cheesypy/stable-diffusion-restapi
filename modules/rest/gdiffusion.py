import numpy as np
import skimage
from skimage.exposure import match_histograms
import PIL

TMP_ROOT_PATH='d:/sd/tmp/'
DEBUG_MODE=False
maskError=0

def _save_debug_img(np_image, name):
    global DEBUG_MODE
    if not DEBUG_MODE: return
    global TMP_ROOT_PATH
    
    image_path = TMP_ROOT_PATH + "/_debug_" + name + ".png"
    if type(np_image) == np.ndarray:
        if np_image.ndim == 2:
            mode = "L"
        elif np_image.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
        pil_image = PIL.Image.fromarray(np.clip(np_image*255., 0., 255.).astype(np.uint8), mode=mode)
        pil_image.save(image_path)
    else:
        np_image.save(image_path)

def extract_mask(init_image):
    mask_image = init_image.split()[-1]
    mask_image = PIL.ImageOps.invert(mask_image)
    init_image = init_image.convert("RGB")
    assert(mask_image.size == init_image.size)
    
    extrema = mask_image.convert("L").getextrema()
    if extrema == (0, 0):
        maskError=1
        mask_image = None  
    elif extrema == (1, 1):
        maskError=2
        mask_image = None  
    else:        
        np_init = (np.asarray(init_image.convert("RGB"))/255.0).astype(np.float64) # annoyingly complex mask fixing
        np_mask_rgb = 1. - (np.asarray(mask_image.convert("RGB"))/255.0).astype(np.float64)
        np_mask_rgb -= np.min(np_mask_rgb)
        np_mask_rgb /= np.max(np_mask_rgb)
        np_mask_rgb = 1. - np_mask_rgb
        _save_debug_img(np_mask_rgb, "np_mask_rgb1")
        np_mask_rgb_hardened = 1. - (np_mask_rgb < 0.99).astype(np.float64)
        blurred = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., channel_axis=2, truncate=32.)
        blurred2 = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., channel_axis=2, truncate=32.)
        #np_mask_rgb_dilated = np_mask_rgb + blurred  # fixup mask todo: derive magic constants
        #np_mask_rgb = np_mask_rgb + blurred
        np_mask_rgb_dilated = np.clip((np_mask_rgb + blurred2) * 0.7071, 0., 1.)
        np_mask_rgb = np.clip((np_mask_rgb + blurred) * 0.7071, 0., 1.)    
        _save_debug_img(np_mask_rgb, "np_mask_rgb2")
        _save_debug_img(np_mask_rgb_dilated, "np_mask_rgb_dilated")

    return np_mask_rgb, np_mask_rgb_dilated, mask_image

def getAlphaAsImage(init_image):
    mask_image = init_image.split()[-1]
    mask_image = PIL.ImageOps.invert(mask_image)
    extrema = mask_image.convert("L").getextrema()
#    print("extrema",extrema)
    if extrema == (0, 0):
        maskError=1
        mask_image = None  
    elif extrema == (1, 1):
        maskError=2
        mask_image = None  

    #if not mask_image.getbbox(): # if mask is all opaque throw error
    #    mask_image = None    
    return mask_image
    
def get_np_init(init_image):
    np_init = (np.asarray(init_image.convert("RGB"))/255.0).astype(np.float64) # annoyingly complex mask fixing
    return np_init



def get_init_image(initimg,mask_blend_factor,noise_q, color_variation):
    width = initimg.size[0]
    height = initimg.size[1]
    np_init=get_np_init(initimg)
    np_mask_rgb, np_mask_rgb_dilated, mask_image=extract_mask(initimg)
    if (not mask_image): return None
    noise_rgb=_get_matched_noise(np_init, np_mask_rgb,noise_q,color_variation)
    blend_mask_rgb = np.clip(np_mask_rgb_dilated,0.,1.) ** (mask_blend_factor)
    noised = noise_rgb[:]
    #noised = ((np_init[:]**1.) ** (1. - blend_mask_rgb)) * ((noise_rgb**(1/1.)))# ** blend_mask_rgb)
    blend_mask_rgb **= (2.)
    noised = np_init[:] * (1. - blend_mask_rgb) + noised * blend_mask_rgb
    
    np_mask_grey = np.sum(np_mask_rgb, axis=2)/3.
    ref_mask =  np_mask_grey < 1e-3
    
    all_mask = np.ones((width, height), dtype=bool)
    noised[all_mask,:] = skimage.exposure.match_histograms(noised[all_mask,:]**1., noised[ref_mask,:], channel_axis=1)                  
    init_image = PIL.Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")

    i=0
    _save_debug_img(init_image, "init_img_" + str(i+1))
    _save_debug_img(mask_image, "mask_img_" + str(i+1))                        
    _save_debug_img(blend_mask_rgb, "blend_mask_rgb_" + str(i+1))

    return init_image, mask_image

# helper fft routines that keep ortho normalization and auto-shift before and after fft
def _fft2(data):
    if data.ndim > 2: # has channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
            out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
    else: # one channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
        out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])
    
    return out_fft
   
def _ifft2(data):
    if data.ndim > 2: # has channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
            out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
    else: # one channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
        out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])
        
    return out_ifft
            
def _get_gaussian_window(width, height, std=3.14, mode=0):

    window_scale_x = float(width / min(width, height))
    window_scale_y = float(height / min(width, height))
    
    window = np.zeros((width, height))
    x = (np.arange(width) / width * 2. - 1.) * window_scale_x
    for y in range(height):
        fy = (y / height * 2. - 1.) * window_scale_y
        if mode == 0:
            window[:, y] = np.exp(-(x**2+fy**2) * std)
        else:
            window[:, y] = (1/((x**2+1.) * (fy**2+1.))) ** (std/3.14) # hey wait a minute that's not gaussian
            
    return window

def _get_masked_window_rgb(np_mask_grey, hardness=1.):
    np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
    if hardness != 1.:
        hardened = np_mask_grey[:] ** hardness
    else:
        hardened = np_mask_grey[:]
    for c in range(3):
        np_mask_rgb[:,:,c] = hardened[:]
    return np_mask_rgb

"""
 Explanation:
 Getting good results in/out-painting with stable diffusion can be challenging.
 Although there are simpler effective solutions for in-painting, out-painting can be especially challenging because there is no color data
 in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.
 Provided here is my take on a potential solution to this problem.
 
 By taking a fourier transform of the masked src img we get a function that tells us the presence and orientation of each feature scale in the unmasked src.
 Shaping the init/seed noise for in/outpainting to the same distribution of feature scales, orientations, and positions increases output coherence
 by helping keep features aligned. This technique is applicable to any continuous generation task such as audio or video, each of which can
 be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased. For multi-channel data such as color
 or stereo sound the "color tone" or histogram of the seed noise can be matched to improve quality (using scikit-image currently)
 This method is quite robust and has the added benefit of being fast independently of the size of the out-painted area.
 The effects of this method include things like helping the generator integrate the pre-existing view distance and camera angle.
 
 Carefully managing color and brightness with histogram matching is also essential to achieving good coherence.
 
 noise_q controls the exponent in the fall-off of the distribution can be any positive number, lower values means higher detail (range > 0, default 1.)
 color_variation controls how much freedom is allowed for the colors/palette of the out-painted area (range 0..1, default 0.01)
 This code is provided as is under the Unlicense (https://unlicense.org/)
 Although you have no obligation to do so, if you found this code helpful please find it in your heart to credit me.
 
 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
 This code is part of a new branch of a discord bot I am working on integrating with diffusers (https://github.com/parlance-zz/g-diffuser-bot)
 
"""
def _get_matched_noise(_np_src_image, np_mask_rgb, noise_q, color_variation): 

    global DEBUG_MODE
    global TMP_ROOT_PATH
    
    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2)/3.) 
    np_src_grey = (np.sum(np_src_image, axis=2)/3.) 
    all_mask = np.ones((width, height), dtype=bool)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3
    
    windowed_image = _np_src_image * (1.-_get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb# / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
    #windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) / (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
    _save_debug_img(windowed_image, "windowed_src_img")
    
    src_fft = _fft2(windowed_image) # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist
    _save_debug_img(src_dist, "windowed_src_dist")
    
    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = np.random.random_sample((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2)/3.) 
    noise_rgb *=  color_variation # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:,:,c] += (1. - color_variation) * noise_grey
        
    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:,:,c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:,:,:] = np.absolute(shaped_noise_fft[:,:,:])**2 * (src_dist ** noise_q) * src_phase # perform the actual shaping
    
    brightness_variation = 0.#color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.
    
    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask,:] = skimage.exposure.match_histograms(shaped_noise[img_mask,:]**1., contrast_adjusted_np_src[ref_mask,:], channel_axis=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb
    _save_debug_img(shaped_noise, "shaped_noise")
    
    matched_noise = np.zeros((width, height, num_channels))
    matched_noise = shaped_noise[:]
    #matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    #matched_noise = _np_src_image[:] * (1. - np_mask_rgb) + matched_noise * np_mask_rgb
    
    _save_debug_img(matched_noise, "matched_noise")
    
    """
    todo:
    color_variation doesnt have to be a single number, the overall color tone of the out-painted area could be param controlled
    """
    
    return np.clip(matched_noise, 0., 1.) 