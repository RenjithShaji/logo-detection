# logo-detection
****Image Similarity Detection using VGG16 and EasyOCR****
This Python script demonstrates a method for detecting image similarity using the pre-trained VGG16 model for feature extraction and EasyOCR for text extraction.

****Dependencies****
`requests: For making HTTP requests.
`BeautifulSoup: For parsing HTML content.
tensorflow: TensorFlow library for machine learning.
numpy: NumPy library for numerical operations.
matplotlib: Matplotlib library for visualization.
scipy: SciPy library for scientific computing.
easyocr: EasyOCR library for text extraction.

****Usage****

Install the required dependencies using pip install -r requirements.txt.
Run the script using Python 3.
Enter the brand name when prompted.
The script will download images related to the entered brand, extract features, and calculate similarity with a test image.
The result (whether the image is real or fake) will be displayed.

****Description*****

download_image(url, keyword, index): Function to download an image from a URL.
search_and_download_images(keyword, max_images): Function to search and download images related to a keyword.
load_and_preprocess_image(image_path): Function to load and preprocess an image for the VGG16 model.
extract_features(image_path): Function to extract features from an image using the VGG16 model.
cosine_similarity(features1, features2): Function to calculate cosine similarity between two feature vectors.
pearson_correlation(features1, features2): Function to calculate the Pearson correlation coefficient between two feature vectors.
extract_text(image_path): Function to extract text from an image using EasyOCR.
calculate_image_similarity(image1_path, image2_path): Function to calculate image similarity using features and text.
delete_downloaded_images(image_paths): Function to delete downloaded images after use.
