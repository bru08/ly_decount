"""
Run inference on test set with a pretrained model
"""

model.eval()

res = []
with torch.no_grad():
  for i, elem in enumerate(test_ds):
    print(f"{i+1}/{len(test_ds)}")

    img = torch.tensor(elem).float().cuda()
    out = model(img.unsqueeze(0)).item()
    res.append((i, out))

len(res)

with open("test_res.csv", "w+") as f:
  f.write("id,count\n")
  for elem in res:
    f.write(f"{elem[0]+1},{round(elem[1])}\n")
