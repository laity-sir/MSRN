import torch
import matplotlib.pyplot as plt

hh=torch.load('./save_data.pth')
print(hh)
train_loss=[]
test_loss=[]
test_psnr=[]
test_ssim=[]
for i in sorted(hh):
    print(i)
    train_loss.append(hh[i]['train_loss'])
    test_loss.append(hh[i]['test_loss'])
    test_psnr.append(hh[i]['test_psnr'])
    test_ssim.append(hh[i]['test_ssim'])
print(train_loss)
print(test_loss)
# plt.plot(train_loss,label='train_loss')
# plt.plot(test_loss,label='test_loss')
# plt.plot(test_psnr,label='test_psnr')
plt.plot(test_ssim,label='test_ssim')
plt.legend(loc='upper right')
plt.title('test_ssim')
plt.savefig('./test_ssim.png')
plt.show()