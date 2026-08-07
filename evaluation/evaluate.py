from evaluation.metrics import count_generated_images

print("=" * 40)
print("GAN EVALUATION REPORT")
print("=" * 40)

total = count_generated_images()

print(f"Generated Images : {total}")

if total > 0:
    print("Status : Synthetic image generation successful.")
else:
    print("Status : No generated images found.")

print("=" * 40)
print("Evaluation Completed Successfully!")