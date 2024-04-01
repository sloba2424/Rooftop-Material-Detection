import cv2
from skimage.transform import resize


class ImageUpscaler:
    _target_resolution = (1200, 1200)

    def __init__(self, model_path='../models/FSRCNN_x4.pb', scale_factor=4):
        # Initialize the DNN Super Resolution object using FSRCNN in the constructor
        self.sr = cv2.dnn_superres.DnnSuperResImpl.create()

        # Read the model
        self.sr.readModel(model_path)

        # Set the desired scale to upscale the image by a factor of 4
        self.sr.setModel('fsrcnn', scale_factor)

    def upscale_image(self, image):
        # Upscale the image
        upscaled_image = self.sr.upsample(image)
        return upscaled_image

    def upscale_image_and_mask(self, image, mask):
        # Upscale the satellite image
        upscaled_image = self.sr.upsample(image)

        # Upscale the mask
        upscaled_mask = resize(mask, self._target_resolution)

        return upscaled_image, upscaled_mask

    @staticmethod
    def resize_and_save_image(image,
                              save_path='../data/upscaling_test/upscaled_image.jpg',
                              target_resolution=(300, 300),
                              include_mask=False,
                              mask=None,
                              mask_save_path='../data/upscaling_test/upscaled_mask.jpg'):
        # Resize the image to the given shape
        resized_image = cv2.resize(image, (target_resolution[1], target_resolution[0]))
        cv2.imwrite(save_path, resized_image)
        print(f"Image saved to {save_path}")

        if include_mask and mask is not None:
            # Resize and save the mask if it's provided and flag is set
            resized_mask = cv2.resize(mask, (target_resolution[1], target_resolution[0]))
            cv2.imwrite(mask_save_path, resized_mask)
            print(f"Mask saved to {mask_save_path}")
            return save_path, mask_save_path

        return save_path


def main():
    # Initialize the ImageUpscaler with the path to your super-resolution model
    iu = ImageUpscaler()

    # Read the image
    image_path = '/Users/Dev/segmentation/src/data/data_1000/train_imgs_1000/000000000012.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Check if image is loaded properly
    if image is None:
        print("Error: Could not read the image.")
    else:
        # Check if the image has 3 channels (Color image)
        if image.shape[2] != 3:
            print("Error: Image is not a color image.")
        else:
            # Upscale the image
            upscaled_image = iu.upscale_image(image)

            # Save the upscaled image using the new method
            # Note: You need to specify the save path if it's different from the default
            iu.resize_and_save_image(upscaled_image, save_path='path_where_to_save/upscaled_image.jpg')


if __name__ == "__main__":
    main()
