from medmnist import PathMNIST

dataset = PathMNIST(split="train", download=True)
print(len(dataset))
print(dataset[0])
