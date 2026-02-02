import os
from glob import glob
from PIL import Image

# Set this to your dataset root
DATASET_ROOT = 'dataset'

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(DATASET_ROOT, split)
    for class_dir in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_file in glob(os.path.join(class_path, '*')):
            try:
                img = Image.open(img_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(img_file)
                    print(f'Converted {img_file} to RGB')
            except Exception as e:
                print(f'Error processing {img_file}: {e}')

print('All images checked and converted to RGB if needed.')
