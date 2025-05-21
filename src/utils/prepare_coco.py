import os
import json
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_coco_dataset(
    coco_dir: str,
    output_dir: str,
    split: str = 'train2017'
):
    """
    Prepare COCO dataset for YOLOv5 training.
    
    Args:
        coco_dir (str): Path to COCO dataset directory
        output_dir (str): Path to output directory
        split (str): Dataset split ('train2017' or 'val2017')
    """
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    images_dir = coco_dir / split
    annotations_file = coco_dir / 'annotations' / f'instances_{split}.json'
    
    # Load annotations
    logger.info(f"Loading annotations from {annotations_file}")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create image ID to filename mapping
    image_id_to_file = {img['id']: img['file_name'] for img in annotations['images']}
    
    # Create category ID to class ID mapping
    cat_id_to_class = {cat['id']: i for i, cat in enumerate(annotations['categories'])}
    
    # Create image ID to annotations mapping
    image_annotations = {}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Process each image
    logger.info(f"Processing {split} images...")
    for img_id, anns in tqdm(image_annotations.items()):
        # Get image filename
        img_file = image_id_to_file[img_id]
        img_path = images_dir / img_file
        
        # Skip if image doesn't exist
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        
        # Create YOLO format annotations
        yolo_anns = []
        for ann in anns:
            # Convert bbox from [x,y,width,height] to [x_center, y_center, width, height] (normalized)
            bbox = ann['bbox']
            img_info = next(img for img in annotations['images'] if img['id'] == img_id)
            img_width, img_height = img_info['width'], img_info['height']
            
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            
            # Get class ID
            class_id = cat_id_to_class[ann['category_id']]
            
            # Add annotation
            yolo_anns.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        # Save image and annotations
        if yolo_anns:  # Only save if there are annotations
            # Copy image
            shutil.copy2(img_path, output_dir / img_file)
            
            # Save annotations
            ann_file = output_dir / f"{img_file.rsplit('.', 1)[0]}.txt"
            with open(ann_file, 'w') as f:
                f.write('\n'.join(yolo_anns))
    
    # Create dataset file list
    with open(output_dir / f"{split}.txt", 'w') as f:
        for img_file in sorted(output_dir.glob('*.jpg')):
            f.write(f"{img_file.relative_to(output_dir.parent)}\n")
    
    logger.info(f"Dataset preparation completed. Output directory: {output_dir}")

def main():
    # Update these paths according to your setup
    coco_dir = "path/to/coco"
    output_dir = "path/to/output"
    
    # Prepare training set
    prepare_coco_dataset(
        coco_dir=coco_dir,
        output_dir=Path(output_dir) / "train2017",
        split='train2017'
    )
    
    # Prepare validation set
    prepare_coco_dataset(
        coco_dir=coco_dir,
        output_dir=Path(output_dir) / "val2017",
        split='val2017'
    )

if __name__ == "__main__":
    main() 