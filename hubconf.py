import pytorch_cifar_models

dependencies = ['torch']

models = filter(lambda name: name.startswith("cifar"), dir(pytorch_cifar_models))
globals().update({model: getattr(pytorch_cifar_models, model) for model in models})
