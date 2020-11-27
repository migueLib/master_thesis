from torchvision import transforms

def get_transform(crop_size=None, resize=(300, 300), grayscale=False, randomcrop=False):
    """
    ORDER IS IMPORTANT HERE
    """
    # Create list of transformations
    transform_list = list()
    
    # Center crop the image
    if crop_size is not None:
        transform_list.append(transforms.CenterCrop(crop_size))
    
    # Vertical and Horizontal flips, affine
    transform_list.append(transforms.RandomVerticalFlip())
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.RandomAffine(90))
    
    # Grayscale
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    
    # Random crop
    if randomcrop:
        transform_list.append(transforms.RandomCrop(size=randomcrop))
        
    # Resize image
    transform_list.append(transforms.Resize(resize))
    
    # Move images to tensors
    transform_list.append(transforms.ToTensor())
    
    # Compose transformations
    TRANSFORM = transforms.Compose(transform_list)
    
    return TRANSFORM