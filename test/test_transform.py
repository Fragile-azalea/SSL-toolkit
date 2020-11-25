from torchvision import transforms as tf
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from allinone import *
    print(TRANSFORM_REGISTRY.catalogue())
    print(tf.Compose(((tf.Resize((32, 32)), tf.ToTensor()))))

    transform = TRANSFORM_REGISTRY('IdentityAndManyTimes')
    transform = transform([tf.ToTensor()], [tf.Resize((32, 32))], 2)
    print(transform)
    transform = TRANSFORM_REGISTRY('ManyTimes')
    transform = transform(tf.Compose([tf.Resize((32, 32)), tf.ToTensor()]), 3)
    print(transform)
    transform = TRANSFORM_REGISTRY('Twice')
    transform = transform(tf.Compose([tf.Resize((32, 32)), tf.ToTensor()]))
    print(transform)

    transform = TRANSFORM_REGISTRY('RandAugment')
    transform = transform(5, 3)
    print(transform)
