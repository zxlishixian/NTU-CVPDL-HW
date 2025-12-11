# CVPDL HW3-DDPM (去噪擴散概率模型) 實現生成手寫數字


### 1. 安裝依賴

### GPU 版本 (CUDA 11.8)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

pip install pillow matplotlib tqdm 

pytorch-fid numpy scipy scikit-learn

### 2. 準備數據集
準備好 MNIST 數據集（MNIST 28×28 RGB）命名為 mnist
（不需要標籤）

./mnist/

  000001.png

  000002.png
  
  ...

### 3. 訓練模型

依次運行cell 1 - 10 開始訓練模型，會在 output 文件夾中產生 best.pt ema_best.pt last.pt ema_last.pt 以及每個epoch訓練完成後產生一張 preview_epochxxx.png 可以用於觀察當前訓練模型效果

### 4. 生成圖像
運行cell 11 可在 generate文件夾中產生10000張模型生成的手寫圖片

>ckpt="/content/drive/MyDrive/Colab Notebooks/cv/hw3/output/ema_best.pt"

注意：一定要使用 ema_best.pt 或者 ema_last.pt 模型權重文件
修改 timesteps=830 的值可以控制擴散步長做適當調整

### 5. 擴散可視化
運行cell 12 可以在 ddpm_mnist 文件夾下生成一張 diffusion_grid.png 用於預先估計模型效果 避免直接生成10000張圖片 耗時過久

### 6. 計算 FID 分數
運行cell 13 - 15 安裝 FID 依賴庫 並將生成的generate文件夾下10000張圖片與 mnist.npz 做比較，得到 FID 分數