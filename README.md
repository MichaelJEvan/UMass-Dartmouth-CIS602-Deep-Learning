Abstract:

Build a small convolutional neural network to count simple shapes: circles,
squares, and triangles in 64×64 grayscale images. A dataset generator within
JupyterNotebook places a random number (0-5) of each shape per image while
preven.ng overlap, and filenames plus ground‑truth counts are saved to CSVs.
The CNN model (SmallCountNet) uses three conv -> ReLU -> max‑pool blocks,
adap.ve average pooling, and a small fully connected head. Training uses the
Adam op.mizer with L1 loss (MAE) and an 80/10/10 train/valida.on/test split; the
final bias is ini.alized to training label means to stabilize learning.
Results show the model predicts counts with very low error: per‑class MAE
typically -0.08 - 0.20 and RMSE similarly small, with most samples predicted
within ±1 of the true count (roughly 85-100% depending on shape). Squares for
some reason are the hardest class to predict. The pipeline includes learning
curves, sca]er and confusion matrices, and error distribu.ons. Next steps are
robustness tests (more overlap, varied sizes/noise) and evalua.on on
out‑of‑distribu.on images.
