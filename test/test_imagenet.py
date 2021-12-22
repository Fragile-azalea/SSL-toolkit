from datetime import date
import unittest

from DeSSL import transforms


class ImageNetTestCase(unittest.TestCase):
    def test_imagenet(self):
        from torchvision import transforms as tf
        from DeSSL.data import semi_imagenet
        from DeSSL import loadding_config
        parser = loadding_config('config/base.yml')
        args = parser.parse_args([])
        imagenet = semi_imagenet(args.root, args.num_labels_per_class, label_transform=tf.RandomResizedCrop(
            (224, 224)), test_transform=[tf.CenterCrop((256, 256)), tf.Resize((224, 224))])
        train_loader, test_loader, classes = imagenet(
            args.batch_size, num_workers=args.num_workers)
        for label, unlabel in train_loader:
            label_data, label_target = label
            unlabel_data, unlabel_target = unlabel
            self.assertEqual(list(label_data.shape), [
                             args.batch_size, 3, 224, 224])
            self.assertEqual(list(unlabel_data.shape), [
                             args.batch_size, 3, 224, 224])
            break
        for data, target in test_loader:
            self.assertEqual(list(data.shape), [
                             args.batch_size * 2, 3, 224, 224])
            break
        assert classes == 1000


if __name__ == '__main__':
    unittest.main()
