from os import listdir
from cv2 import imread, IMREAD_COLOR, IMREAD_UNCHANGED, IMREAD_GRAYSCALE, getRotationMatrix2D, add, subtract, INTER_CUBIC, flip, filter2D, GaussianBlur, resize
from numpy import array, ones, float64, mean, zeros 

class ImageProcessing:
    def __init__(self, config):
        color_mode_mapping = {
            "unchanged" : IMREAD_UNCHANGED,
            "color" : IMREAD_COLOR,
            "gray_scale" : IMREAD_GRAYSCALE 
        }
        if config["image_processing_color_mode"] in color_mode_mapping.keys():
            color_flag = color_mode_mapping[config["image_processing_color_mode"]]
        else:
            color_flag = IMREAD_UNCHANGED
        self.images = self.read_images(config["dataset_directory"], color_flag)
        self.config = config
    def read_images(self, directory, color_flag):
        images = []
        for file in listdir(directory):
            images.append(imread(directory + file, color_flag))
        return images
    def Rotate(self, image):
        rotation = self.config["image_augmentation_parameters"]["rotation"]
        steps = self.config["image_augmentation_parameters"]["steps"]
        images=[]
        for i in range(-rotation,rotation+1,steps):
            rotation_matrix=getRotationMatrix2D(center=(image.shape[1]/2,image.shape[2]/2),
            angle=i, scale=1)
            rotated_image=warpAffine(src=image, M=rotation_matrix, 
            dsize=(image.shape[1], image.shape[0]))
            images.append(rotated_image)
        return array(images)
    def AdjustBrightness(self, image):
        images=[]
        mask=ones(image.shape,dtype='uint8') * self.config["image_augmentation_parameters"]["brightness_mask_constant"]
        images.append(add(image,mask))
        images.append(subtract(image,mask))
        return images
    def FlipImage(self, image):
        images=[]
        modes=[-1,0,1]
        for i in modes:
            images.append(flip(image,i))
        return images

    def Sharpening(self, image):
        kernel=array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])  # Laplacian Kernel
        image=filter2D(src=image,ddepth=-1,kernel=kernel)
        return image

    def Smoothing(self, image):
        image=GaussianBlur(image,(self.config["image_augmentation_parameters"]["gaussian_smoothing_filter_size"][0],self.config["image_augmentation_parameters"]["gaussian_smoothing_filter_size"][1]),0)
        return image

    def Resize(self, image):
        dimension = (self.config["image_processing_image_size"][0], self.config["image_processing_image_size"][1])
        image=resize(image,dimension,interpolation=INTER_CUBIC)
        return image
    def main(self):
        self.images = list(map(self.Resize, self.images))
        if self.config["is_augment_image"] == True:
            augmention_fns = [
                self.Rotate,
                self.AdjustBrightness,
                self.FlipImage,
                self.Sharpening,
                self.Smoothing
            ]
            augmentations = []
            for augmentation in augmention_fns:
                augmentations.extend(list(map(augmentation, self.images)))
            self.images.extend(augmentations)
        return self.images

    
