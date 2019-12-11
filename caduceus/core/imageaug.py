from imgaug import augmenters as iaa
import tensorflow as tf
import numpy as np
# # Standard scenario: You have N RGB-images and additionally 21 heatmaps per image.
# # You want to augment each image and its heatmaps identically.
# images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
# heatmaps = np.random.randint(0, 255, (16, 128, 128, 21), dtype=np.uint8)

# seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(translate_px={"x": (-40, 40)})])

# # Convert the stochastic sequence of augmenters to a deterministic one.
# # The deterministic sequence will always apply the exactly same effects to the images.
# seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
# images_aug = seq_det.augment_images(images)
# heatmaps_aug = seq_det.augment_images(heatmaps)

def imgaugmentation(BatchSrc, BatchTar):

    BatchSrc = BatchSrc*255.0
    BatchTar = BatchTar*255.0
    # BatchSrc = np.expand_dims(BatchSrc, 2)
    # BatchTar = np.expand_dims(BatchTar, 2)
    noVec = BatchSrc.shape[-1]
    BatchSrc_aug = np.zeros(BatchSrc.shape)
    BatchTar_aug = np.zeros(BatchTar.shape)
    #seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(translate_px={"x": (-40, 40)})])
    seq = iaa.Sequential([
    #iaa.Fliplr(0.5), # horizontal flips
    # iaa.Crop(percent=(0, 0.1)), # random crops
    # # Small gaussian blur with random sigma between 0 and 0.5.
    # # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.Sometimes(0.5, 
        iaa.ContrastNormalization((0.75, 1.5))
    ),
    # # Add gaussian noise.
    # # For 50% of all images, we sample the noise once per pixel.
    # # For the other 50% of all images, we sample the noise per pixel AND
    # # channel. This can change the color (not only brightness) of the
    # # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # # Make some images brighter and some darker.
    # # In 20% of all cases, we sample the multiplier once per channel,
    # # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # # Apply affine transformations to each image.
    # # Scale/zoom them, translate/move them, rotate them and shear them.
    # iaa.Affine(
    #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     rotate=(-25, 25),
    #     shear=(-8, 8)
    #)

    # Sharpen each image, overlay the result with the original
    # image using an alpha between 0 (no sharpening) and 1
    # (full sharpening effect).
    iaa.Sometimes(0.5, 
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
    ),
    ], random_order=True) # apply augmenters in random order
    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
    for g in range(0, noVec):
        print 'g = ' + `g`
        BatchSrc_aug[:,:,:,:,g] = seq_det.augment_images(BatchSrc[:,:,:,:,g])
        BatchTar_aug[:,:,:,:,g] = seq_det.augment_images(BatchTar[:,:,:,:,g])
    # BatchSrc_aug = np.squeeze(BatchSrc_aug, 2)
    # BatchTar_aug = np.squeeze(BatchTar_aug, 2)
    BatchSrc_aug = BatchSrc_aug/255.0
    BatchTar_aug = BatchTar_aug/255.0
    return BatchSrc_aug, BatchTar_aug

# def imgaugmentation(BatchSrc, BatchTar):
    
#     BatchSrc = BatchSrc*255.0
#     BatchTar = BatchTar*255.0
#     # BatchSrc = np.expand_dims(BatchSrc, 2)
#     # BatchTar = np.expand_dims(BatchTar, 2)
#     noVec = BatchSrc.shape[-1]

#     #seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(translate_px={"x": (-40, 40)})])
#     seq = iaa.Sequential([
#     #iaa.Fliplr(0.5), # horizontal flips
#     # iaa.Crop(percent=(0, 0.1)), # random crops
#     # # Small gaussian blur with random sigma between 0 and 0.5.
#     # # But we only blur about 50% of all images.
#     iaa.Sometimes(0.5,
#         iaa.GaussianBlur(sigma=(0, 0.5))
#     ),
#     # Strengthen or weaken the contrast in each image.
#     iaa.Sometimes(0.5, 
#         iaa.ContrastNormalization((0.75, 1.5))
#     ),
#     # # Add gaussian noise.
#     # # For 50% of all images, we sample the noise once per pixel.
#     # # For the other 50% of all images, we sample the noise per pixel AND
#     # # channel. This can change the color (not only brightness) of the
#     # # pixels.
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#     # # Make some images brighter and some darker.
#     # # In 20% of all cases, we sample the multiplier once per channel,
#     # # which can end up changing the color of the images.
#     iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     # # Apply affine transformations to each image.
#     # # Scale/zoom them, translate/move them, rotate them and shear them.
#     # iaa.Affine(
#     #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#     #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#     #     rotate=(-25, 25),
#     #     shear=(-8, 8)
#     #)

#     # Sharpen each image, overlay the result with the original
#     # image using an alpha between 0 (no sharpening) and 1
#     # (full sharpening effect).
#     iaa.Sometimes(0.5, 
#         iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
#     ),
#     ], random_order=True) # apply augmenters in random order
#     seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
#     BatchSrc_aug = seq_det.augment_images(BatchSrc)
#     BatchTar_aug = seq_det.augment_images(BatchTar)
#     # BatchSrc_aug = np.squeeze(BatchSrc_aug, 2)
#     # BatchTar_aug = np.squeeze(BatchTar_aug, 2)
#     BatchSrc_aug = BatchSrc_aug/255.0
#     BatchTar_aug = BatchTar_aug/255.0
#     return BatchSrc_aug, BatchTar_aug
    