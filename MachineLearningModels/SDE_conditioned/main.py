import os
import json
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import optim
from SDE_train import ModelTrainer
from SDE_datareduction import get_data
from SDE_test import calculate_metrics, sample_save_metrics
from SDE_utils import set_seed, load_images
from script_util import create_model_diffusion

DEFAULT_SEED = 42

# Base Paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
print(f"Script Dir {SCRIPT_DIR}")
# BASE_OUTPUT = r"/home/20234635/MachineLearningGit/MachineLearningModels/SDE_conditioned/results"
BASE_OUTPUT = "results"

# BASE_INPUT = r"C:\Users\tabor\Documents\Programming\MachineLearning\data"
BASE_INPUT = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

# Dataset paths
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_specific")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_combined")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_fixrange")
DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_maxrange_5000")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
STRUCTURE_DATASET_PATH = os.path.join(DATASET_PATH, "Structure")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = device == "cuda"

TEST_PATH = r"/home/20234635/MachineLearningGit/MachineLearningModels/SDE_conditioned/results/UNet_nblocks_2_noisesteps_250_smartsplit_False_8793/"
SAMPLE_MODEL = "best_model.pth"

def gen_run_name(config):
    """
    Generate a unique run name for this training instance.
    Args:
        config: dict with values for this configuration.

    Returns:
        run_name: str that contains the run name for this training instance.
        run_inst: int with the specific id for this training instance.
    """
    # Generate a 4-digit random job ID
    run_inst = random.randint(1000, 9999)  # Generates a random integer between 1000 and 9999
    run_name_base = f"{config['model_name']}_nblocks_{config['n_blocks']}_noisesteps_{config['noise_steps']}_smartsplit_{config['smart_split']}"
    run_name = f"{run_name_base}_{run_inst}"

    # Check for existing run names and increment if needed (but now by generating a new random id)
    while os.path.exists(os.path.join(BASE_OUTPUT, run_name)):
        run_inst = random.randint(1000, 9999)  # generate a new random ID
        run_name = f"{run_name_base}_{run_inst}"

    return run_name, run_inst

def main():
    with open('config.json', "r", encoding="utf-8") as f:
        config = json.load(f)

    if config['training'] is True and config['testing'] is not True:

        run_name, run_inst = gen_run_name(config)

        # Output paths
        result_path = os.path.join(BASE_OUTPUT, run_name)
        model_path = os.path.join(result_path, "models")
        image_path = os.path.join(result_path, "images")

        set_seed(seed=DEFAULT_SEED)
        print(f"Name: {run_name}")

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # Initialize model and diffusion classes with parameters from config
        model, diffusion = create_model_diffusion(device, **config)
        # AdamW optimizer for increased generalization with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

        train_dataloader, val_dataloader, test_dataloader, *_ = get_data(image_dataset_path=IMAGE_DATASET_PATH,
                                                                              structure_dataset_path=STRUCTURE_DATASET_PATH,
                                                                              result_path=result_path, **config)
        # Initialize trainer class
        trainer = ModelTrainer(model=model,
                               device=device,
                               optimizer=optimizer,
                               image_path=image_path,
                               model_path=model_path,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               test_dataloader=test_dataloader,
                               diffusion=diffusion,
                               **config)
        # Start training
        trainer.train()

        torch.save(trainer.best_model_checkpoint, os.path.join(model_path, "best_model.pth"))

        # If EMA was used, different model has to be saved
        if trainer.ema is True and trainer.ema_model is not None:
            torch.save(trainer.ema_model.state_dict(), os.path.join(model_path, "ema_model.pth"))

        max_ssim = max(trainer.ssim_values)
        print(f"Max SSIM: {max_ssim}, At place: {5 * np.argmax(trainer.ssim_values)+4}")
        min_mae = min(trainer.mae_values)
        print(f"Min MAE: {min_mae}, At place: {5 * np.argmin(trainer.mae_values)+4}")
        print(f"Best Model Epoch: {trainer.best_model_epoch}")

        np.savez(os.path.join(result_path, "train_losses.npz"), losses=trainer.train_losses)
        np.savez(os.path.join(result_path, "val_losses.npz"), losses=trainer.val_losses)
        np.savez(os.path.join(result_path, "ssim_values.npz"), losses=trainer.ssim_values)
        np.savez(os.path.join(result_path, "mae_values.npz"), losses=trainer.mae_values)

        plt.figure(figsize=(12, 6))
        plt.plot(trainer.train_losses[1:], label='Train Loss')
        plt.plot(trainer.val_losses[1:], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Loss over Epochs')
        plt.savefig(os.path.join(result_path, "losses.png"))
        plt.show()

        x_values = range(0, len(trainer.ssim_values) * 5, 5)
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, trainer.ssim_values, label='SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('SSIM over Epochs')
        plt.savefig(os.path.join(result_path, "SSIM.png"))

        x_values = range(0, len(trainer.mae_values) * 5, 5)
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, trainer.mae_values, label='MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE over Epochs')
        plt.savefig(os.path.join(result_path, "MAE.png"))

        if config['generate_images']:

            if config['ema']:
                model = trainer.ema_model
            else:
                model.load_state_dict(trainer.best_model_checkpoint)

            model_results = sample_save_metrics(model=model,
                                                device=device,
                                                sampler=trainer.diffusion,
                                                test_dataloader=test_dataloader,
                                                result_path=result_path, **config)

            model_results['train id'] = run_inst
            model_results['bm ssim'] = max_ssim
            model_results['bm mae'] = min_mae
            model_results['bm epoch'] = trainer.best_model_epoch

            df_model_results = pd.DataFrame(model_results)

            excel_results = os.path.join(BASE_OUTPUT, "results.xlsx")

            try:
                with pd.ExcelWriter(excel_results, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    df_model_results.to_excel(writer, sheet_name='Results', index=False, header=False,
                                              startrow=writer.sheets[
                                                  'Results'].max_row if 'Results' in writer.sheets else 0)
            except FileNotFoundError:
                print(f"Error: The file '{excel_results}' was not found. Creating a new file.")
                with pd.ExcelWriter(excel_results, mode='w', engine='openpyxl') as writer:
                    df_model_results.to_excel(writer, sheet_name='Results', index=False)

    if config['testing']:
        parameter_path = os.path.join(TEST_PATH, 'parameters.json')
        model_path = os.path.join(os.path.join(TEST_PATH, "models"), SAMPLE_MODEL)
        image_path = os.path.join(TEST_PATH, "images")
        sample_path = os.path.join(image_path, "Samples")
        reference_path = os.path.join(image_path, "References")
        test_dataset_path = os.path.join(TEST_PATH, "test_indices.pth")

        if config['calculate_metrics']:
            reference_images = load_images(reference_path)
            sampled_images = load_images(sample_path)
            ssim_values, psnr_values, mae_values, max_error_values = calculate_metrics(reference_images, sampled_images)
            print(f"SSIM: {np.mean(ssim_values)}, "
                  f"PSNR: {np.mean(psnr_values)}, "
                  f"MAE: {np.mean(mae_values)}, "
                  f"Max Error: {np.max(max_error_values)}")

        if config['sample_metrics']:
            with open(parameter_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            print(f"Sampling model: {model_path}")
            set_seed(seed=DEFAULT_SEED)
            model, sampler = create_model_diffusion(device, **config)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            sample_save_metrics(model=model,
                                device=device,
                                sampler=sampler,
                                test_path=test_dataset_path,
                                result_path=TEST_PATH, **config)
            return


if __name__ == "__main__":
    test_splits = [0.1]
    for _ in range(1):
        for test_split in test_splits:
            with open("config.json", "r+", encoding="utf-8") as file:
                config = json.load(file)
                config['test_split'] = test_split
                config['smart_split'] = False
                file.seek(0)
                json.dump(config, file, indent=4)
                file.truncate()

            main()
