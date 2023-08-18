#### My own implementation of augmentations using Albumentations in combination with the YOLOv8 model by ultralytics



class Albumentations:
    # YOLOv8 Albumentations class (optional, only used if package is installed)
    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A

            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            T = [
                
                A.Rotate(p=0.2, limit=10),
                A.Perspective(p=0.2),
                A.HorizontalFlip(p=0.5),

                A.MotionBlur(p=0.33, blur_limit=(3,37)),
                
                A.RandomToneCurve(p=0.2),
                A.Sharpen(p=0.2),
                A.RandomGamma(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2, hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20),
                A.CLAHE(p=0.2),

                A.OneOf([
                    A.ISONoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.GaussNoise(p=1, per_channel=False, var_limit=(10, 100)),
                    A.GaussNoise(p=1, per_channel=True, var_limit=(10, 100))],
                p=0.33)
                
                ]  # transforms
            
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels['img']
        cls = labels['cls']
        if len(cls):
            labels['instances'].convert_bbox('xywh')
            labels['instances'].normalize(*im.shape[:2][::-1])
            bboxes = labels['instances'].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new['class_labels']) > 0:  # skip update if no bbox in new im
                    labels['img'] = new['image']
                    labels['cls'] = np.array(new['class_labels'])
                    bboxes = np.array(new['bboxes'])
            labels['instances'].update(bboxes=bboxes)
        return labels