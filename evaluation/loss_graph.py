import matplotlib.pyplot as plt

generator_losses = [1.2, 1.0, 0.9, 0.8, 0.7]
discriminator_losses = [0.8, 0.7, 0.6, 0.6, 0.5]

epochs = range(1, len(generator_losses) + 1)

plt.plot(
    epochs,
    generator_losses,
    label="Generator Loss"
)

plt.plot(
    epochs,
    discriminator_losses,
    label="Discriminator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")

plt.legend()

plt.savefig(
    "evaluation/gan_training_loss.png"
)

plt.show()