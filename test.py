import ujson
import matplotlib.pyplot as plt

path_to_conf = "train/configs/sum2win_losses.json"

with open(path_to_conf, "r") as file:
  losses = ujson.load(file)
  fig, axs = plt.subplots(2, 2, figsize=(15, 10))

  axs[0, 0].plot(losses['GA_Loss'], label='Gen_A')
  axs[0, 0].plot(losses['GB_Loss'], label='Gen_B')
  axs[0, 0].set_title("Generator Losses")
  axs[0, 0].legend()

  axs[0, 1].plot(losses['DA_Loss'], label='Dis_A')
  axs[0, 1].plot(losses['DB_Loss'], label='Dis_B')
  axs[0, 1].set_title("Discriminator Losses")
  axs[0, 1].legend()

  axs[1, 0].plot(losses['PSNR_avg'], label='PSNR')
  axs[1, 0].plot(losses['Cycle_PSNR_avg'], label='Cycle_PSNR')
  axs[1, 0].set_title("PSNR Metrics")
  axs[1, 0].legend()

  axs[1, 1].plot(losses['SSIM_avg'], label='SSIM')
  axs[1, 1].plot(losses['Cycle_SSIM_avg'], label='Cycle_SSIM')
  axs[1, 1].set_title("SSIM Metric")
  axs[1, 1].legend()

  plt.tight_layout()
  plt.savefig("1.png")