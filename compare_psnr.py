with open('val_set_info.txt') as f:
   psnr_before = []
   for l in f.readlines():
       idx=l.find('psnr')
       if idx==-1:
           continue
       psnr_before.append(l[idx+6:idx+11])

with open('training_fixedqp.log') as f:
   psnr_after = []
   for l in f.readlines():
       idx = l.find('psnr')
       if idx == -1:
           continue
       psnr_after.append(l[idx+6:idx+11])

print(psnr_before)
print(psnr_after)

diff = [float(psnr_after[i]) - float(psnr_before[i]) for i in range(len(psnr_before))]
for i in range(len(diff)):
    print(f'idx {i}, psnr gain {diff[i]:.4f}')
