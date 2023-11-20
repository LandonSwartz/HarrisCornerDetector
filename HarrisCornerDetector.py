import cv2
import numpy as np


class HarrisCornerDetector:
    def __init__(self, window_size=5, k=0.04):
        self.window_size = window_size
        self.k = k
        self.image = None
        self.gray_image = None
        self.blurred_image = None
        self.Ix = None
        self.Iy = None
        self.R = None

    def load_image(self, image_path):
        # load the image from the image path
        self.image = cv2.imread(image_path)
        # convert to grayscale
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self):
        self.blurred_image = cv2.GaussianBlur(
            self.gray_image, (self.window_size, self.window_size), 0
        )

    def compute_gradients(self):
        self.Ix = cv2.Sobel(
            self.blurred_image, cv2.CV_64F, 1, 0, ksize=self.window_size
        )
        self.Iy = cv2.Sobel(
            self.blurred_image, cv2.CV_64F, 0, 1, ksize=self.window_size
        )

    def detect_corners(self):
        Ix2 = self.Ix**2
        Iy2 = self.Iy**2
        IxIy = self.Ix * self.Iy

        offset = self.window_size // 2
        height, width = self.gray_image.shape
        self.R = np.zeros((height, width), dtype=np.float64)

        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                Sx2 = np.sum(
                    Ix2[y - offset : y + offset + 1, x - offset : x + offset + 1]
                )
                Sy2 = np.sum(
                    Iy2[y - offset : y + offset + 1, x - offset : x + offset + 1]
                )
                Sxy = np.sum(
                    IxIy[y - offset : y + offset + 1, x - offset : x + offset + 1]
                )

                detM = (Sx2 * Sy2) - (Sxy**2)
                traceM = Sx2 + Sy2

                self.R[y, x] = detM - self.k * (traceM**2)

    def mark_corners(self, threshold_ratio=0.1):
        threshold = threshold_ratio * self.R.max()
        corner_image = np.copy(self.image)
        offset = self.window_size // 2
        height, width = self.gray_image.shape

        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                if self.R[y, x] > threshold:
                    cv2.circle(corner_image, (x, y), 5, (255, 0, 0), -1)

        return corner_image

    def apply_non_maximal_suppression(self, neighborhood_size=3):
        height, width = self.R.shape
        offset = neighborhood_size // 2
        suppressed_R = np.zeros((height, width), dtype=np.float64)

        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                local_max = np.max(
                    self.R[y - offset : y + offset + 1, x - offset : x + offset + 1]
                )
                if self.R[y, x] == local_max:
                    suppressed_R[y, x] = self.R[y, x]

        self.R = suppressed_R

    def find_corners(self, image_path, threshold_ratio=0.1):
        self.load_image(image_path)
        self.apply_gaussian_blur()
        self.compute_gradients()
        self.detect_corners()
        self.apply_non_maximal_suppression()
        corner_image = self.mark_corners(threshold_ratio)

        return corner_image
