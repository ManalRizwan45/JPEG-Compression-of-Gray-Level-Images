JPEG Compression of Gray Level Images

This Python project demonstrates an image compression and processing pipeline using Bit Plane Slicing and Discrete Cosine Transform (DCT) techniques. The code is built using libraries such as OpenCV, NumPy, SciPy, Matplotlib, and Tkinter.

Features:

Load a grayscale image.
Ensure image dimensions are multiples of 8 for DCT processing.
Apply level shifting to center pixel values.
Use DCT for frequency domain transformation.
Quantize and dequantize image coefficients.
Implement Bit Plane Slicing for encoding and decoding.
Enhance image contrast using histogram equalization.
Display images and key data with a graphical user interface.
Prerequisites:

Python 3.x
Required libraries: OpenCV, NumPy, SciPy, Matplotlib, Tkinter, PIL (Python Imaging Library)
Install libraries using: pip install opencv-python numpy scipy matplotlib pillow

Usage:

Clone the repository.
Install required libraries.
Run image_processing.py to execute the image processing pipeline.
The GUI will show original, processed, and final images.
Results:

The pipeline showcases the following images:

Original image
Level-shifted image
Quantized image
DCT-transformed image
Encoded image (Bit Plane Slicing)
Decoded image (Bit Plane Slicing)
Dequantized image
Image after inverse DCT transformation
Image after reversing level shifting
Image after histogram equalization
The GUI displays original size, compressed size, decompressed size, and PSNR (Peak Signal-to-Noise Ratio) information.
