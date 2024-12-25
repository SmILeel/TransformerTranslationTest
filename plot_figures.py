import matplotlib.pyplot as plt
import torch

results = torch.load('./cache/metrics_record.pkl')

# 假设你已经有了这些结果
iter_losses = results['iter_loss']
epoch_losses = results['epoch_loss']
train_accs = results['train_acc']
val_accs = results['val_acc']


# 设置统一的字体和图表大小
plt.rcParams.update({'font.family': 'Times New Roman', 'figure.figsize': (8, 6)})

# 1. 绘制每个iteration下的损失
plt.figure()
plt.plot(range(len(iter_losses)), iter_losses, label='Iteration Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Iteration Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./cache/iter_loss.png')
plt.show()

# 2. 绘制每个epoch的损失
plt.figure()
plt.plot(range(len(epoch_losses)), epoch_losses, label='Epoch Loss', color='red')
# plt.plot(range(len(val_accs)), val_accs, label='Val Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./cache/epoch_loss.png')
plt.show()

# 3. 绘制每个epoch的验证机准确率
plt.figure()
plt.plot(range(len(val_accs)), val_accs, label='Val Accuracy', color='red')
# plt.plot(range(len(val_accs)), val_accs, label='Val Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./cache/val_acc.png')
plt.show()


# 4. 绘制每个epoch下的训练精度与验证精度
plt.figure()
plt.plot(range(len(train_accs)),  train_accs, label='Validation Accuracy', color='red')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Train Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./cache/train_val_acc.png')
plt.show()

