# segmenting_cnn
A Convolutional Neural Network for automatic semantic segmentation of confocal microscopy images.

Written to handle 2D images of confluent tissues (where plasma membrane markers clearly and brightly label the borders of each cell), but can be adapted to take 3D image volumes as input.

4 scripts currently included:

(1) unet_model: establishes network architecture to be used while training. 
    
    Current specs -- 4 down-sampling encoder blocks, convolution, then 4 up-sampling decoder blocks, so that output masks are the same size as input.

(2) unet_train: uses the architecture in unet_model to train a model, based on ground truth images and masks.
   
    Required 'inputs' -- paths to folder containing correspondingly named images and masks (i.e. 'image{i}' matches 'mask{i}'); NUM_IMAGES = number of images/ masks in training set
    Tunable parameters -- NUM_EPOCHS = number of iterations through training set, reduce for faster processing; BATCH_SIZE = number of images oer batch, reduce for faster processing.

(3) unet_segment(image_path, checkpoint_path): loads desired model and segments a single image. 
    
    Required inputs -- image_path = directory containing image to be segmented; checkpoint_path = directory containing trained model

(4) unet_segment_movie(images_path, checkpoint_path, numZ): loads desired model and segments every image contained in images_path. Handles each Z-plane as an individual 2D image, with no information shared between slices of the same volume (this is definitely an area that could be improved by a fully 3D approach!)
    
    Required inputs -- image_path = directory containing image to be segmented; checkpoint_path = directory containing trained model; numZ = number of Z-slices in image aquisition.
    Fucntion assumes each time point and Z-slice are saved individually, with naming convention: frame{i:0>{4}}_z{z:0>{2}}
    
