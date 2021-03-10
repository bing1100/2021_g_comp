class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        print(dataset)
        if dataset == 'pascal':
            return '/media/bhux/ssd/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'dse':
            return '/media/bhux/ssd/grss_dse'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
