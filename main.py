import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_PATH = 'image.jpg'
D0_VALUES = [5, 10, 50, 250]
D0_PAIRS = [(5, 10), (50, 250)]
BUTTERWORTH_N = 2

SAVE_FOLDER = 'results'
os.makedirs(SAVE_FOLDER, exist_ok=True)

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
cv2.imwrite('GRAYSCALE.jpg', image)

def to_uint8_image(img):
    img = np.abs(img)
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max() * 255
    return img.astype(np.uint8)


def create_spectrum_image(fshift):
    spectrum = 20 * np.log(np.abs(fshift) + 1)
    return to_uint8_image(spectrum)


def create_distance_matrix(rows, cols):
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    center_row = rows // 2
    center_col = cols // 2
    D = np.sqrt((U - center_row) ** 2 + (V - center_col) ** 2)
    return D



def create_ideal_low_pass_filter(D, D0):
    H = np.zeros_like(D, dtype=np.float32)
    H[D <= D0] = 1
    return H


def create_butterworth_low_pass_filter(D, D0, n=2):
    H = 1 / (1 + (D / D0) ** (2 * n))
    return H.astype(np.float32)


def create_gaussian_low_pass_filter(D, D0):
    H = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
    return H.astype(np.float32)


def create_ideal_high_pass_filter(D, D0):
    H = np.ones_like(D, dtype=np.float32)
    H[D <= D0] = 0
    return H


def create_butterworth_high_pass_filter(D, D0, n=2):
    eps = 1e-6
    H = 1 / (1 + (D0 / (D + eps)) ** (2 * n))
    return H.astype(np.float32)


def create_gaussian_high_pass_filter(D, D0):
    H = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))
    return H.astype(np.float32)


def save_combined_filter_results(filter_name, filter_func, D, fshift, d0_values, save_path):
    fig, axes = plt.subplots(len(d0_values), 3, figsize=(10, 6))

    for row, D0 in enumerate(d0_values):
        H = filter_func(D, D0)
        filtered_shift, filtered_img = apply_frequency_filter(fshift, H)

        images = [
            ('Kernel', H),
            ('Filtered spectrum', create_spectrum_image(filtered_shift)),
            ('Restored', filtered_img),
        ]

        for col, (title, image) in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image, cmap='gray')
            ax.set_title(f'{title}, D0 = {D0}')
            ax.axis('off')

    fig.suptitle(filter_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def save_combined_3d_filters(filter_name, filter_func, D, d0_values, save_path):
    step = max(1, min(D.shape) // 100)
    y = np.arange(0, D.shape[0], step)
    x = np.arange(0, D.shape[1], step)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 5))

    for index, D0 in enumerate(d0_values, start=1):
        H = filter_func(D, D0)[::step, ::step]
        ax = fig.add_subplot(1, len(d0_values), index, projection='3d')
        ax.plot_surface(X, Y, H, cmap='viridis', linewidth=0, antialiased=True)
        ax.set_title(f'D0 = {D0}')
        ax.set_xlabel('v')
        ax.set_ylabel('u')
        ax.set_zlabel('H(u, v)')
        ax.set_zlim(0, 1)

    fig.suptitle(f'{filter_name} 3D')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)



def apply_frequency_filter(fshift, H):
    filtered_shift = fshift * H
    filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    filtered_img = np.abs(filtered_img)
    return filtered_shift, filtered_img


filters = [
    ('low_frequency', 'ideal_low', create_ideal_low_pass_filter),
    ('low_frequency', 'butterworth_low', lambda D, D0: create_butterworth_low_pass_filter(D, D0, BUTTERWORTH_N)),
    ('low_frequency', 'gaussian_low', create_gaussian_low_pass_filter),
    ('high_frequency', 'ideal_high', create_ideal_high_pass_filter),
    ('high_frequency', 'butterworth_high', lambda D, D0: create_butterworth_high_pass_filter(D, D0, BUTTERWORTH_N)),
    ('high_frequency', 'gaussian_high', create_gaussian_high_pass_filter),
]

rows, cols = image.shape
D = create_distance_matrix(rows, cols)



fft = np.fft.fft2(image)
fshift = np.fft.fftshift(fft)
spectrum_image = create_spectrum_image(fshift)
cv2.imwrite('spectrum_image.jpg', spectrum_image)


#main
for frequency_folder, filter_name, filter_func in filters:
    output_folder = os.path.join(SAVE_FOLDER, frequency_folder)
    os.makedirs(output_folder, exist_ok=True)

    for d0_values in D0_PAIRS:
        d0_suffix = f'D0_{d0_values[0]}_{d0_values[1]}'

        save_combined_filter_results(
            filter_name,
            filter_func,
            D,
            fshift,
            d0_values,
            os.path.join(output_folder, f'{filter_name}_{d0_suffix}_results.png'),
        )
        save_combined_3d_filters(
            filter_name,
            filter_func,
            D,
            d0_values,
            os.path.join(output_folder, f'{filter_name}_{d0_suffix}_3d.png'),
        )
