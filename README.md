# AuraSR: GAN Super-Resolution for Images 🖼️

[![Replicate](https://replicate.com/zsxkib/aura-sr/badge)](https://replicate.com/zsxkib/aura-sr)

AuraSR is a powerful tool that makes images bigger and clearer. It's based on the [GigaGAN](https://mingukkang.github.io/GigaGAN/) idea and works great for certain types of images.

![See AuraSR in action](https://storage.googleapis.com/falserverless/gallery/aurasr-animated.webp)

## Quick Start 🚀

Use Cog to make your image bigger:
```bash
cog predict -i image=@your_image.png -i scale_factor=4
```

## What It Does 🎨

- Makes PNG, lossless WebP, and high-quality JPEG XL (90+) images bigger and clearer
- Can make images 2, 4, 8, 16, or 32 times bigger
- Works fast and can handle different sized jobs

## Things to Know ⚠️

AuraSR is great, but it has some limits:

1. It works best with PNG, lossless WebP, and high-quality JPEG XL (90+) images.
2. It doesn't like images that have been squeezed too much (compressed).
3. It can't fix mistakes in images.
4. It's best for making AI-generated images or very high-quality photos bigger.

## What You Need 📋

- Python 3.7 or newer
- Cog

## How to Use It 🛠️

### Simple Way
```bash
cog predict -i image=@your_image.png -i scale_factor=4
```

### Advanced Way
```bash
cog predict -i image=@your_image.png -i scale_factor=8 -i max_batch_size=4
```

## Options 🔧

- `image`: The picture you want to make bigger
- `scale_factor`: How much bigger you want to make it (2, 4, 8, 16, or 32)
- `max_batch_size`: Controls how fast it works (default is 1, increase if you have a powerful computer)

## Thank You 🙌

- [GAN-based Super-Resolution](https://github.com/fal-ai/aura-sr/tree/main) for real-world images, a variation of the GigaGAN paper for image-conditioned upscaling. PyTorch implementation is based on the unofficial [lucidrains/gigagan-pytorch](https://github.com/lucidrains/gigagan-pytorch) repository.

## Let's Talk 🐦

Have questions? Follow me on Twitter [@zsakib_](https://twitter.com/zsakib_) and let's chat!