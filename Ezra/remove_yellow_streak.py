from PIL import Image, ImageOps, ImageFilter

# input a PIL image and the function will return that image with the yellow removed
# there are two steps to the process: a) removing the yellow, b) filling in the whited out image
def remove_yellow(img):
    # Convert the image to RGB mode
    img = img.convert("RGB")

    # Get the pixel data
    pixels = img.load()

    # Define the color as yellow
    color_to_remove = (255, 255, 0)

    # color distance threshold
    threshold = 180

    # Define the color distance function
    def color_distance(c1, c2):
        return sum([(x-y)**2 for x, y in zip(c1, c2)])**0.5

    changed = False
    # Iterate through all pixels
    for x in range(img.width):
        for y in range(img.height):
            # Check if the color distance between the pixel color and the color to remove is less than the threshold
            if color_distance(pixels[x, y], color_to_remove) < threshold:
                # print("changing color")
                # Set the pixel color to white 
                pixels[x, y] = (255, 255, 255)
                changed = True
    
    # If the image is unchanged, because it has no yellow, then continue
    # But if the image was changed, then fill in whited out portions
    if (changed):
        # convert to grayscale
        img.convert('L')

         # Invert the image
        img = ImageOps.invert(img)

        # Define the structuring element and perform morphological closing
        kernel = Image.new('1', (4, 4), 1)
        img = img.filter(ImageFilter.MinFilter(1))
        img = img.filter(ImageFilter.MaxFilter(1))

        # Invert the image back
        img = ImageOps.invert(img)
    
    return img