import cv2
import fingerprint_feature_extractor
from fingerprint_enhancer.fingerprint_image_enhancer import FingerprintImageEnhancer
import sys
import time

def save_minutiae_to_file(file_path, features_terminations, features_bifurcations):
    with open(file_path, "w") as f:
        # Process Terminations
        for minutiae in features_terminations:
            x = minutiae.locX
            y = minutiae.locY
            angle = minutiae.Orientation if minutiae.Orientation >= 0 else minutiae.Orientation + 360
            type_code = 1  # Termination
            f.write(f"{x},{y},{angle},{type_code}\n")

        # Process Bifurcations
        for minutiae in features_bifurcations:
            x = minutiae.locX
            y = minutiae.locY
            angle = minutiae.Orientation if minutiae.Orientation >= 0 else minutiae.Orientation + 360
            type_code = 2  # Bifurcation
            f.write(f"{x},{y},{angle},{type_code}\n")

def to_iso19794(img_shape, minutiae, output_path):
    width, height = img_shape
    b_array = bytearray()
    minutiae_num = len(minutiae)
    total_bytes = minutiae_num * 6 + 28 + 2

    # Header (Format Identifier "FMR\0" and version "20\0")
    b_array = bytearray(b'FMR\x00 20\x00')
    
    # Total bytes (4 bytes, big endian)
    b_array += total_bytes.to_bytes(4, 'big')
    
    # Reserved bytes
    b_array += bytearray(b'\x00\x00')
    
    # Image dimensions
    b_array += width.to_bytes(2, 'big')
    b_array += height.to_bytes(2, 'big')
    
    # Resolution and other parameters
    b_array += bytearray(b'\x00\xc5\x00\xc5\x01\x00\x00\x00d')
    
    # Number of minutiae (1 byte)
    b_array += minutiae_num.to_bytes(1, 'big')

    # Process each minutia
    for minutia in minutiae:
        x = int(minutia.locX)
        y = int(minutia.locY)
        angle = int(minutia.Orientation[0])
        min_type = int(minutia.Type)
        quality = int(90)

        # Ensure values stay within byte ranges
        # First byte: type (2 bits) + x high bits (6 bits)
        byte0 = (min_type << 6) | ((x >> 8) & 0x3F)
        
        # Second byte: x low bits
        byte1 = x & 0xFF
        
        # Third byte: y high bits
        byte2 = (y >> 8) & 0xFF
        
        # Fourth byte: y low bits
        byte3 = y & 0xFF
        
        # Fifth byte: angle (normalized to 0-255)
        byte4 = int((360 - angle) * 256 / 360) % 256
        
        # Sixth byte: quality
        byte5 = quality & 0xFF

        # Add bytes to array
        b_array += bytes([byte0, byte1, byte2, byte3, byte4, byte5])

    # Extended data (empty in this case)
    b_array += (0).to_bytes(2, 'big')

    # Write to file
    with open(output_path, 'wb') as istfile:
        istfile.write(b_array)

if __name__ == "__main__":
    image_enhancer = FingerprintImageEnhancer()  # Create object called image_enhancer
    IMG_NAME = "101_1.tif"
    img = cv2.imread("./images/" + IMG_NAME)

    if len(img.shape) > 2:  # convert image into gray if necessary
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_enhanced = image_enhancer.enhance(img, invert_output=True)  # run image enhancer
    image_enhancer.save_enhanced_image(f"output/enhanced/{IMG_NAME.split('.')[0]}.jpg")  # save enhanced image

    img = (255 * img_enhanced).astype("uint8")

    # get img width and height
    img_shape = (img.shape[1], img.shape[0])

    FeaturesTerminations, FeaturesBifurcations = (
        fingerprint_feature_extractor.extract_minutiae_features(
            IMG_NAME,
            img,
            spuriousMinutiaeThresh=10,
            invertImage=True,
            showResult=False,
            saveResult=True,
        )
    )
    Features = FeaturesTerminations + FeaturesBifurcations

    to_iso19794(img_shape, Features, f'./output/ISO/{IMG_NAME.split(".")[0]}.iso')

    print("Minutiae features extracted successfully!")