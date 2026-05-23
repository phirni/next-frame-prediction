This project compares two deep learning architectures for next-frame prediction on Atari 
Pong gameplay. ConvLSTM achieves superior reconstruction quality, while β-VAE+RNN 
delivers 10× faster inference suitable for real-time applications. 
Intially i wanted to implemented gamengen paper, but due to constraints I had concluded on this idea on implementing it on atari ping pong.

Key Finding: ConvLSTM wins on accuracy metrics (PSNR, SSIM, MSE, MAE), but 
β-VAE+RNN excels in inference speed (752 FPS vs 77 FPS) and perceptual quality (LPIPS).

Source: OpenAI Gym Atari Pong-v0 
Total Frames: 3,000 frames (~50 seconds of gameplay) 
Resolution: 210×160 (original) → 64×64 (processed) 
Format: Grayscale, normalized [0,1]

Key Insights 
ConvLSTM Strengths: 
● Superior pixel-level accuracy (lower MSE/MAE) 
● Better structural similarity (higher SSIM) 
● Cleaner visual predictions with less blur 
β-VAE+RNN Strengths: 
● 10× faster inference (critical for real-time applications) 
● Better perceptual quality (lower LPIPS despite higher MSE) 
● Compact 64-dim latent space enables efficient processing 

Trade-off Analysis 
● Accuracy vs Speed: ConvLSTM sacrifices speed for ~37% better MSE; 
β-VAE+RNN sacrifices accuracy for 10× speed gain 
● Perceptual Paradox: β-VAE+RNN achieves better LPIPS despite worse MSE, 
suggesting latent compression preserves perceptually important features 
● Memory Parity: Both models use ~3.9GB, indicating the speed difference stems 
from architectural efficiency, not resource usage 
