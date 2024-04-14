import os
import cv2
import numpy as np
import tqdm as tqdm
import utils as utils

def add_salt_and_pepper_noise(image, noise_amount):
    # Generate random values for each pixel
    random_values = np.random.rand(*image.shape[:2])

    # Add salt noise
    image[random_values < noise_amount] = 255

    # Add pepper noise
    image[random_values > 1 - noise_amount] = 0


def calculate_psnr(original_frame, noisy_frame):
    # normalize the pixel values to [0, 1]
    original_frame = original_frame / 255.0
    noisy_frame = noisy_frame / 255.0

    # calculate the MSE
    mse = np.mean(np.power((original_frame - noisy_frame), 2))
    print(mse)

    # calculate the PSNR
    psnr = -10 * np.log10(mse)
    print(psnr)
    return psnr

def apply_salt_and_pepper_noise_to_frames(input_folder, output_folder, noise_amount):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    psnr_sum = 0
    num_frames = 0

    for filename in tqdm.tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        frame = cv2.imread(input_path)
        noisy_frame = frame.copy()
        add_salt_and_pepper_noise(noisy_frame, noise_amount)

        cv2.imwrite(output_path, noisy_frame)

        psnr = calculate_psnr(frame, noisy_frame)
        psnr_sum += psnr
        num_frames += 1

    average_psnr = psnr_sum / num_frames
    print("Average PSNR:", average_psnr)
    print("Salt and pepper noise applied to frames and saved in the target folder.")

# Example usage
input_folder = "data/honeybee"
output_folder = "data/honeybee_noisy"
noise_amount = 0.0026  # Adjust this value to control the amount of noise

apply_salt_and_pepper_noise_to_frames(input_folder, output_folder, noise_amount)