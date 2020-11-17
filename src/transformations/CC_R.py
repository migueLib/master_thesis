from torchvision import transforms

def get_transform(crop_size, resize=(299, 299), grayscale=False):
    # Create list of transformations
    transform_list = list()
    
    # Center crop the image
    if crop_size is not None:
        transform_list.append(transforms.CenterCrop(crop_size))
        
    # Resize
    transform_list.append(transforms.Resize(resize))
    
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    
    # Send to tensor
    transform_list.append(transforms.ToTensor())
    
    # Compose transformations
    TRANSFORM = transforms.Compose(transform_list)
    
    return TRANSFORM